"""Load Arabic visual-speech transcripts and prepare camera-compatible crops."""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_preprocessor import MIN_FACE_DETECTION_RATE, VISPER_ALIGNMENT_IDS, extract_face_points
from visper_adapter import file_sha256
from visper_inference import MOUTH_SIZE, mouth_images


SPLITS = ("train", "validation", "test")
FPS = 25
MAX_SECONDS = 8
MIN_FRAMES = 10
MAX_FRAMES = MAX_SECONDS * FPS


@dataclass(frozen=True)
class Clip:
    path: Path
    text: str
    speaker: str
    split: str
    digest: str


def training_text(text: str) -> str:
    characters = (
        char if char.isalnum() else " "
        for char in unicodedata.normalize("NFKC", text)
        if char != "ـ" and not unicodedata.category(char).startswith("M")
    )
    return " ".join("".join(characters).split())


def arabic_transcript(text: str) -> str:
    text = training_text(text)
    if not any(char.isalpha() and "ARABIC" in unicodedata.name(char, "") for char in text):
        raise ValueError("transcript must contain Arabic letters")
    return text


class VisperTokenizer:
    def __init__(self, source: Path, decoder):
        import sentencepiece

        self.processor = sentencepiece.SentencePieceProcessor(
            model_file=str(source / "spm" / "unigram" / "unigram.model")
        )
        self.units = {token: index for index, token in enumerate(decoder.tokens)}
        self.unknown = self.units["<unk>"]
        self.language = decoder.language

    def encode(self, text: str) -> list[int]:
        tokens = [self.units.get(piece, self.unknown) for piece in self.processor.encode(text, out_type=str)]
        if not tokens or self.unknown in tokens:
            raise ValueError("the ViSpeR tokenizer cannot represent this transcript")
        return [self.language, *tokens]


def ctc_minimum_frames(tokens: list[int]) -> int:
    return len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))


def load_manifest(path: Path) -> dict[str, list[Clip]]:
    splits: dict[str, list[Clip]] = {split: [] for split in SPLITS}
    speakers: dict[str, str] = {}
    seen: set[str] = set()
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            clip = (path.parent / row["clip"]).resolve(strict=True)
            text = arabic_transcript(row["text"])
            speaker = row["speaker"].strip().casefold()
            split = row["split"]
            if not speaker or split not in SPLITS:
                raise ValueError("clip requires Arabic text, speaker and a valid split")
            if speaker in speakers and speakers[speaker] != split:
                raise ValueError(f"speaker {speaker!r} appears in multiple splits")
            digest = file_sha256(clip)
            if digest in seen:
                raise ValueError("identical clip already appears in the manifest")
            speakers[speaker] = split
            seen.add(digest)
            splits[split].append(Clip(clip, text, speaker, split, digest))
        except (KeyError, AttributeError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{index}: {error}") from error
    for split, clips in splits.items():
        if not clips:
            raise ValueError(f"Manifest needs clips in the {split} split")
    return splits


def video_crops(path: Path, landmarker, reference: np.ndarray) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames, anchors, times = [], [], []
    try:
        rate = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(rate) or rate < 12:
            raise ValueError("video must be readable and have at least 12 frames per second")
        if capture.get(cv2.CAP_PROP_FRAME_COUNT) > math.ceil(rate * MAX_SECONDS):
            raise ValueError(f"trim the clip to at most {MAX_SECONDS} seconds")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if len(frames) >= math.ceil(rate * MAX_SECONDS):
                raise ValueError(f"trim the clip to at most {MAX_SECONDS} seconds")
            height, width = frame.shape[:2]
            scale = min(1.0, 1280 / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            anchors.append(extract_face_points(frame, landmarker, VISPER_ALIGNMENT_IDS))
    finally:
        capture.release()

    if not frames:
        raise ValueError("video has no readable frames")
    times = np.asarray(times)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        times = np.arange(len(frames)) / rate
    else:
        times -= times[0]
    if times[-1] >= MAX_SECONDS:
        raise ValueError(f"trim the clip to at most {MAX_SECONDS} seconds")
    targets = np.arange(math.floor(times[-1] * FPS + 1e-6) + 1) / FPS
    valid_indices = [i for i, points in enumerate(anchors) if points is not None]
    if len(valid_indices) < max(2, MIN_FACE_DETECTION_RATE * len(frames)):
        raise ValueError("the face was missing or unclear in too many frames")
    valid_times = times[valid_indices]
    if valid_times[0] > 0.08 or times[-1] - valid_times[-1] > 0.08 or np.any(np.diff(valid_times) > 0.16):
        raise ValueError("the face tracking has a long gap")
    output = mouth_images(frames, times, anchors, targets, reference)
    if output is None:
        raise ValueError("mouth alignment failed")
    return output


def load_crops(path: Path, landmarker, reference: np.ndarray) -> np.ndarray:
    crops = (
        np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npy" else video_crops(path, landmarker, reference)
    )
    if (
        crops.dtype != np.uint8 or crops.ndim != 3 or crops.shape[1:] != (MOUTH_SIZE, MOUTH_SIZE)
        or not MIN_FRAMES <= len(crops) <= MAX_FRAMES
    ):
        raise ValueError(f"expected {MIN_FRAMES}–{MAX_FRAMES} mouth frames as uint8 [T, 96, 96]")
    return crops
