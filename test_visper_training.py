import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from camera_collection import CameraCollection
from camera_personalization import SCHEMA, CameraPersonalization, SavedFeedbackError
from train_visper import score
from visper_adapter import adapter_weights, install_adapters, merge_weights
from visper_training_data import load_manifest


class TrainingTests(unittest.TestCase):
    def test_collection_resumes_and_rejects_a_changed_speaker_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root / "arabic.txt"
            prompts.write_text("متى ترجع؟\nبعد قليل\n", encoding="utf-8")
            session = CameraCollection(root, prompts, "S1", "train")
            crops = np.zeros((25, 96, 96), dtype=np.uint8)
            session.save(crops)
            row = json.loads(session.manifest.read_text(encoding="utf-8"))
            self.assertEqual(row["text"], "متى ترجع؟")
            self.assertTrue(np.array_equal(np.load(root / row["clip"], allow_pickle=False), crops))
            resumed = CameraCollection(root, prompts, "s1", "train")
            self.assertEqual(resumed.prompt, "بعد قليل")
            with self.assertRaisesRegex(ValueError, "another split"):
                CameraCollection(root, prompts, "s1", "test")

    def test_manifest_detects_recordings_shared_between_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "manifest.jsonl"
            rows = []
            for index, split in enumerate(("train", "validation", "test")):
                clip = root / f"{split}.npy"
                clip.write_bytes(bytes([index]))
                rows.append({"clip": clip.name, "speaker": split, "split": split, "text": "أين الطريق"})

            def write_manifest():
                path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")

            write_manifest()
            self.assertEqual(len(load_manifest(path)["test"]), 1)
            rows[-1]["speaker"] = "Train"
            write_manifest()
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                load_manifest(path)
            rows[-1]["speaker"] = "test"
            (root / rows[-1]["clip"]).write_bytes(b"\x00")
            write_manifest()
            with self.assertRaisesRegex(ValueError, "identical clip"):
                load_manifest(path)

    def test_fused_adapter_matches_trained_layers_and_preserves_base(self):
        attention = nn.Module()
        attention.linear_q = nn.Linear(5, 5)
        attention.linear_v = nn.Linear(5, 5)
        layer = nn.Module()
        layer.src_attn = attention
        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.decoders = nn.ModuleList([layer])
        model.ctc = nn.Module()
        model.ctc.ctc_lo = nn.Linear(5, 7)
        model.encoder = nn.Linear(5, 5)
        adapters = install_adapters(model, rank=2)
        projection = model.ctc.ctc_lo
        before = projection.layer.weight.detach().clone()
        self.assertFalse(model.encoder.weight.requires_grad)
        self.assertFalse(projection.layer.weight.requires_grad)
        sample = torch.ones(3, 5)
        projection(sample).square().mean().backward()
        with torch.no_grad():
            projection.up.weight.add_(projection.up.weight.grad, alpha=-0.1)
        adapted = projection(sample)
        for name, adapter in adapters.items():
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, adapter.layer)
        merge_weights(model, adapter_weights(adapters))
        torch.testing.assert_close(model.ctc.ctc_lo(sample), adapted)
        self.assertFalse(torch.equal(model.ctc.ctc_lo.weight, before))

    def test_word_error_accounts_for_empty_and_wrong_predictions(self):
        self.assertIsNone(score([])["wer"])
        result = score([(["كلمة"], []), (["سأصل", "غدا"], ["سأصل", "اليوم"])])
        self.assertAlmostEqual(result["wer"], 2 / 3)
        self.assertEqual(result["reference_words"], 3)


class PersonalTrainingTests(unittest.TestCase):
    def test_replays_same_text_and_rotates_words_that_have_waited_longest(self):
        learner = CameraPersonalization(None, Path("."), Path("base.pt"), None, Path("feedback.sqlite3"))
        learner.profile = "test"
        learner.through_sequence = 6
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript(SCHEMA)
            labels = ["أين", "وصلت", "قريب", "الآن", "انتظر", "أين"]
            db.executemany(
                "INSERT INTO feedback(sequence, profile, attempt, text, corrected, encoded) VALUES (?, ?, ?, ?, ?, ?)",
                [(i, learner.profile, str(i), label, i != 5, bytes([i])) for i, label in enumerate(labels, 1)],
            )
            db.executemany(
                "INSERT INTO feedback_replay(profile, text, last_sequence) VALUES (?, ?, ?)",
                [("test", "وصلت", 3), ("test", "قريب", 4)],
            )
            first = learner.select_replay(db, "أين")
            self.assertEqual(first, [("أين", b"\x06"), ("الآن", b"\x04"), ("انتظر", b"\x05")])
            db.executemany(
                "INSERT INTO feedback_replay(profile, text, last_sequence) VALUES (?, ?, ?)",
                [("test", text, 7) for text, _ in first],
            )
            second = learner.select_replay(db, "بعد")
            self.assertEqual([text for text, _ in second[:2]], ["وصلت", "قريب"])

    def test_saves_both_feedback_kinds_and_recovers_interrupted_training(self):
        def make_session(database):
            attention = nn.Module()
            attention.linear_q = nn.Linear(4, 4)
            attention.linear_v = nn.Linear(4, 4)
            layer = nn.Module()
            layer.src_attn = attention
            model = nn.Module()
            model.decoder = nn.Module()
            model.decoder.decoders = nn.ModuleList([layer])
            model.ctc = nn.Module()
            model.ctc.ctc_lo = nn.Linear(4, 3)
            decoder = SimpleNamespace(model=model, device="cpu")
            learner = CameraPersonalization(decoder, database.parent, database, None, database)
            learner.profile = "test"
            learner.tokenizer = SimpleNamespace(encode=lambda text: [2, 1])
            return learner

        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "feedback.sqlite3"
            learner = make_session(database)
            features = torch.ones((6, 4))
            with self.assertRaisesRegex(ValueError, "letters or numbers"):
                learner.save_feedback("invalid", features, " ", True)
            self.assertFalse(database.exists())
            with patch.object(learner, "fit", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(SavedFeedbackError):
                    learner.save_feedback("one", features, "أين الطريق؟", True)

            resumed = make_session(database)

            def fit(target, replay):
                with torch.no_grad():
                    resumed.adapters["ctc.ctc_lo"].up.weight.fill_(0.25)
                return True

            with patch.object(resumed, "fit", side_effect=fit) as update:
                self.assertTrue(resumed.resume().updated)
                progress = resumed.save_feedback("two", features, "٢", False)
                self.assertTrue(progress.updated)
                self.assertIs(progress.applied, True)
                self.assertEqual((progress.confirmed, progress.corrected), (1, 1))
                self.assertFalse(resumed.save_feedback("two", features, "٢", False).updated)
                self.assertEqual(update.call_count, 2)
                self.assertEqual(update.call_args.args[1][0][0], "أين الطريق")

            with patch.object(resumed, "fit", side_effect=RuntimeError("interrupted again")):
                with self.assertRaises(SavedFeedbackError):
                    resumed.save_feedback("pending", features, "وصلت", True)
            with patch.object(resumed, "fit", side_effect=[True, False]) as update:
                rejected = resumed.save_feedback("repeat", features, "٢", True)
                self.assertTrue(rejected.updated)
                self.assertIs(rejected.applied, False)
                self.assertIn("٢", [text for text, _ in update.call_args.args[1]])
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(db.execute("SELECT text, corrected FROM feedback ORDER BY sequence").fetchall(),
                                 [("أين الطريق", 1), ("٢", 0), ("وصلت", 1), ("٢", 1)])
            reloaded = make_session(database)
            self.assertEqual(reloaded.resume("repeat").saved, 4)
            self.assertIs(reloaded.resume("repeat").applied, False)
            weight = reloaded.adapters["ctc.ctc_lo"].up.weight
            torch.testing.assert_close(weight, torch.full_like(weight, 0.25))

    def test_pair_comparison_uses_one_encoding_and_restores_the_adapter(self):
        encoded = torch.ones(3, 4)
        adapter = SimpleNamespace(enabled=True)
        seen = []

        def transcribe(features):
            self.assertIs(features, encoded)
            seen.append(adapter.enabled)
            return "بعد" if adapter.enabled else "قبل"

        decoder = SimpleNamespace(transcribe_encoded=transcribe)
        learner = CameraPersonalization(decoder, Path("."), Path("base.pt"), None, Path("feedback.sqlite3"))
        learner.adapters["layer"] = adapter
        self.assertEqual(learner.compare_predictions(encoded), ("بعد", "قبل"))
        self.assertEqual(seen, [True, False])
        self.assertTrue(adapter.enabled)

        def fail_baseline(_):
            if not adapter.enabled:
                raise RuntimeError("baseline failed")
            return "بعد"

        decoder.transcribe_encoded = fail_baseline
        with self.assertRaises(RuntimeError):
            learner.compare_predictions(encoded)
        self.assertTrue(adapter.enabled)


if __name__ == "__main__":
    unittest.main()
