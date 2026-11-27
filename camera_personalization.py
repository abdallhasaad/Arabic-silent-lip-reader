"""Incremental visual-only learning from feedback on a camera attempt."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch

from train_visper import adaptation_loss
from visper_adapter import adapter_weights, file_sha256, install_adapters, restore_adapter_weights
from visper_training_data import VisperTokenizer, ctc_minimum_frames, training_text


SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    sequence INTEGER PRIMARY KEY,
    profile TEXT NOT NULL,
    attempt TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    corrected INTEGER NOT NULL,
    encoded BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_profile ON feedback(profile, sequence);
CREATE TABLE IF NOT EXISTS feedback_update (
    sequence INTEGER PRIMARY KEY REFERENCES feedback(sequence),
    applied INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback_replay (
    profile TEXT NOT NULL,
    text TEXT NOT NULL,
    last_sequence INTEGER NOT NULL,
    PRIMARY KEY(profile, text)
);
CREATE TABLE IF NOT EXISTS personal_adapter (
    profile TEXT PRIMARY KEY,
    through_sequence INTEGER NOT NULL,
    weights BLOB
);
"""
RANK = 8
UPDATE_STEPS = 2
REPLAY_SIZE = 3
LEARNING_RATE = 0.0003


@dataclass(frozen=True)
class LearningProgress:
    saved: int
    updated: bool
    confirmed: int = 0
    corrected: int = 0
    applied: bool | None = None


class SavedFeedbackError(RuntimeError):
    """Feedback committed successfully and its training can be resumed."""

    def __init__(self, message: str, corrected: bool):
        super().__init__(message)
        self.corrected = corrected


def to_blob(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def from_blob(blob: bytes, device="cpu"):
    return torch.load(io.BytesIO(blob), map_location=device, weights_only=True)


class CameraPersonalization:
    def __init__(self, decoder, source: Path, checkpoint: Path, adapter: Path | None, database: Path):
        self.decoder = decoder
        self.source = source
        self.checkpoint = checkpoint
        self.base_adapter = adapter
        self.database = database
        self.adapters = {}
        self.through_sequence = 0

    @cached_property
    def tokenizer(self):
        try:
            return VisperTokenizer(self.source, self.decoder)
        except ImportError as error:
            raise RuntimeError("Install requirements.txt to enable camera personalization") from error

    @cached_property
    def profile(self) -> str:
        files = [
            self.checkpoint,
            self.source / "spm" / "unigram" / "unigram.model",
            self.source / "spm" / "unigram" / "unigram_units.txt",
            self.source / "data_prepare" / "20words_mean_face.npy",
        ]
        if self.base_adapter is not None:
            files.append(self.base_adapter)
        digest = "|".join(file_sha256(path) for path in files)
        return hashlib.sha256(f"visual-feedback-v1|{digest}".encode("ascii")).hexdigest()

    def save_feedback(self, attempt: str, encoded: torch.Tensor, text: str, corrected: bool) -> LearningProgress:
        text = training_text(text)
        if not text:
            raise ValueError("feedback needs a transcript with letters or numbers")
        if ctc_minimum_frames(self.tokenizer.encode(text)) > len(encoded):
            raise ValueError("transcript is too long for this camera attempt")
        stored = encoded.to(device="cpu", dtype=torch.float16)
        if not torch.isfinite(stored).all():
            raise ValueError("invalid visual features in this camera attempt")
        profile = self.profile
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database, timeout=30)) as db, db:
            db.executescript(SCHEMA)
            db.execute(
                "INSERT INTO feedback(profile, attempt, text, corrected, encoded) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(attempt) DO NOTHING",
                (profile, attempt, text, corrected, to_blob(stored)),
            )
        try:
            return self.resume(attempt)
        except Exception as error:
            raise SavedFeedbackError(str(error), corrected) from error

    def refresh(self, db: sqlite3.Connection) -> bytes | None:
        state = db.execute(
            "SELECT through_sequence, weights FROM personal_adapter WHERE profile = ?",
            (self.profile,),
        ).fetchone()
        if state is None:
            return None
        through_sequence, weights = state
        if through_sequence > self.through_sequence:
            if weights is not None:
                self.ensure_adapters()
                restore_adapter_weights(self.adapters, from_blob(weights))
            self.through_sequence = through_sequence
        return weights

    def refresh_before_prediction(self) -> None:
        if self.database.is_file():
            with closing(sqlite3.connect(self.database, timeout=30)) as db:
                self.refresh(db)

    def compare_predictions(self, encoded: torch.Tensor) -> tuple[str, str]:
        personalized = self.decoder.transcribe_encoded(encoded)
        if not self.adapters:
            return personalized, personalized
        for adapter in self.adapters.values():
            adapter.enabled = False
        try:
            baseline = self.decoder.transcribe_encoded(encoded)
        finally:
            for adapter in self.adapters.values():
                adapter.enabled = True
        return personalized, baseline

    def resume(self, attempt: str | None = None) -> LearningProgress:
        if not self.database.is_file():
            return LearningProgress(0, False)
        updated = False
        with closing(sqlite3.connect(self.database, timeout=30)) as db:
            with db:
                db.executescript(SCHEMA)
            while True:
                applied = self.learn_next(db)
                if applied is None:
                    break
                updated = applied or updated
            saved, corrected = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(corrected), 0) FROM feedback WHERE profile = ?",
                (self.profile,),
            ).fetchone()
            outcome = None
            if attempt is not None:
                outcome = db.execute(
                    "SELECT u.applied FROM feedback AS f JOIN feedback_update AS u USING(sequence) "
                    "WHERE f.profile = ? AND f.attempt = ?",
                    (self.profile, attempt),
                ).fetchone()
        applied = bool(outcome[0]) if outcome is not None else None
        return LearningProgress(saved, updated, saved - corrected, corrected, applied)

    def learn_next(self, db: sqlite3.Connection) -> bool | None:
        # Serialize pending work across app windows. Commit weights and progress together.
        original = None
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                weights = self.refresh(db)
                row = db.execute(
                    "SELECT sequence, text, encoded FROM feedback WHERE profile = ? AND sequence > ? "
                    "ORDER BY sequence LIMIT 1",
                    (self.profile, self.through_sequence),
                ).fetchone()
                if row is None:
                    return None
                sequence, text, encoded = row
                replay = self.select_replay(db, text)
                self.ensure_adapters()
                original = adapter_weights(self.adapters)
                applied = self.fit((text, encoded), replay)
                if applied:
                    weights = to_blob(adapter_weights(self.adapters))
                else:
                    restore_adapter_weights(self.adapters, original)
                db.executemany(
                    "INSERT INTO feedback_replay(profile, text, last_sequence) VALUES (?, ?, ?) "
                    "ON CONFLICT(profile, text) DO UPDATE SET last_sequence=excluded.last_sequence",
                    [(self.profile, replay_text, sequence) for replay_text, _ in replay],
                )
                db.execute("INSERT INTO feedback_update(sequence, applied) VALUES (?, ?)", (sequence, applied))
                db.execute(
                    "INSERT INTO personal_adapter(profile, through_sequence, weights) VALUES (?, ?, ?) "
                    "ON CONFLICT(profile) DO UPDATE SET through_sequence=excluded.through_sequence, weights=excluded.weights",
                    (self.profile, sequence, weights),
                )
        except Exception:
            if original is not None:
                restore_adapter_weights(self.adapters, original)
            raise
        self.through_sequence = sequence
        return applied

    def select_replay(self, db: sqlite3.Connection, text: str) -> list[tuple[str, bytes]]:
        # Keep one past recording per text, then rehearse texts least recently used.
        return db.execute(
            """
            WITH examples AS (
                SELECT MAX(sequence) AS sequence FROM feedback
                WHERE profile = ? AND sequence <= ? GROUP BY text
            )
            SELECT f.text, f.encoded
            FROM examples JOIN feedback AS f USING(sequence)
            LEFT JOIN feedback_replay AS r ON r.profile = f.profile AND r.text = f.text
            ORDER BY (f.text = ?) DESC, COALESCE(r.last_sequence, 0), f.corrected DESC, f.sequence
            LIMIT ?
            """,
            (self.profile, self.through_sequence, text, REPLAY_SIZE),
        ).fetchall()

    def ensure_adapters(self) -> None:
        if not self.adapters:
            self.adapters = install_adapters(self.decoder.model, RANK)
            self.decoder.model.eval()

    def fit(self, target, replay) -> bool:
        samples = [
            (from_blob(blob, self.decoder.device).float(), self.tokenizer.encode(text))
            for text, blob in (target, *replay)
        ]
        parameters = [parameter for parameter in self.decoder.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
        fractions = [0.65, *(0.35 / len(replay) for _ in replay)] if replay else [1.0]

        with torch.no_grad():
            before = [adaptation_loss(self.decoder.model, *sample).item() for sample in samples]
        for _ in range(UPDATE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            for sample, fraction in zip(samples, fractions):
                (adaptation_loss(self.decoder.model, *sample) * fraction).backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(gradient):
                optimizer.zero_grad(set_to_none=True)
                return False
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            after = [adaptation_loss(self.decoder.model, *sample).item() for sample in samples]
        # Favor the new transcript while guarding against loss increases on past phrases.
        return after[0] < before[0] and sum(after[1:]) <= 1.05 * sum(before[1:])
