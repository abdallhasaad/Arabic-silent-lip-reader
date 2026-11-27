"""Adapt the visual ViSpeR model and compare it on speaker-disjoint Arabic clips."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from camera_evaluation import word_edits, word_tokens
from camera_preprocessor import create_face_landmarker
from visper_adapter import adapter_weights, file_sha256, install_adapters, merge_weights
from visper_inference import ASSETS, VISPER_SOURCE, VisperDecoder, default_checkpoint
from visper_training_data import Clip, VisperTokenizer, ctc_minimum_frames, load_crops, load_manifest


@dataclass
class EncodedClip:
    clip: Clip
    encoding: Path
    tokens: list[int]


def encode_dataset(decoder, splits, source: Path, task: Path, cache: Path):
    try:
        tokenizer = VisperTokenizer(source, decoder)
    except ImportError as error:
        raise RuntimeError("Install requirements-training.txt before training") from error

    videos = any(clip.path.suffix.lower() != ".npy" for clips in splits.values() for clip in clips)
    cached = {}
    with create_face_landmarker(task) if videos else nullcontext() as landmarker:
        for split, clips in splits.items():
            cached[split] = []
            for index, clip in enumerate(clips, 1):
                try:
                    tokens = tokenizer.encode(clip.text)
                    encoding = decoder.encode(load_crops(clip.path, landmarker, decoder.reference)).cpu().contiguous()
                    if len(encoding) < ctc_minimum_frames(tokens):
                        raise ValueError("transcript is too long for the available frames")
                    if not torch.isfinite(encoding).all():
                        raise ValueError("visual features contain nonfinite values")
                    path = cache / f"{clip.digest}.pt"
                    torch.save(encoding, path)
                    cached[split].append(EncodedClip(clip, path, tokens))
                except (OSError, ValueError) as error:
                    raise ValueError(f"{clip.path}: {error}") from error
                if index == len(clips) or index % 100 == 0:
                    print(f"Encoded {split} clips {index}/{len(clips)}", flush=True)
    return cached


def adaptation_loss(model, encoding, tokens: list[int]):
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
    from espnet.nets.pytorch_backend.transformer.mask import target_mask

    features = encoding.unsqueeze(0)
    targets = torch.tensor([tokens], device=features.device, dtype=torch.long)
    lengths = torch.tensor([features.shape[1]], dtype=torch.long)
    ctc_loss, _ = model.ctc(features, lengths, targets)
    decoder_in, decoder_out = add_sos_eos(targets, model.sos, model.eos, model.ignore_id)
    logits, _ = model.decoder(decoder_in, target_mask(decoder_in, model.ignore_id), features, None)
    # The language is supplied to the decoder at inference, as in upstream training.
    attention_loss = model.criterion(logits[:, 1:].contiguous(), decoder_out[:, 1:].contiguous())
    return model.mtlalpha * ctc_loss + (1 - model.mtlalpha) * attention_loss


def score(rows):
    errors = chars_wrong = words = chars = 0
    for reference, prediction in rows:
        errors += word_edits(reference, prediction)
        words += len(reference)
        reference_chars = list("".join(reference))
        prediction_chars = list("".join(prediction))
        chars_wrong += word_edits(reference_chars, prediction_chars)
        chars += len(reference_chars)
    return {
        "clips": len(rows), "reference_words": words, "reference_characters": chars,
        "word_errors": errors, "character_errors": chars_wrong,
        "wer": errors / words if words else None, "cer": chars_wrong / chars if chars else None,
    }


def evaluate(decoder, clips: list[EncodedClip], training_vocabulary: set[str]):
    decoder.model.eval()
    decoder.search.eval()
    rows = []
    for sample in clips:
        encoding = torch.load(sample.encoding, map_location=decoder.device, weights_only=True)
        prediction = decoder.transcribe_encoded(encoding)
        rows.append((word_tokens(sample.clip.text), word_tokens(prediction)))
    return {
        "overall": score(rows),
        "single_words": score([row for row in rows if len(row[0]) == 1]),
        "phrases": score([row for row in rows if len(row[0]) > 1]),
        "contains_unseen_word": score([row for row in rows if set(row[0]) - training_vocabulary]),
    }


def train(decoder, cached, args):
    training_vocabulary = {
        word for sample in cached["train"] for word in word_tokens(sample.clip.text)
    }
    baseline_validation = evaluate(decoder, cached["validation"], training_vocabulary)
    print(f"Baseline validation WER {baseline_validation['overall']['wer']:.1%}", flush=True)
    adapters = install_adapters(decoder.model, args.rank)
    parameters = [p for p in decoder.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    examples = cached["train"].copy()
    random_order = random.Random(args.seed)
    best = None
    history = []

    for epoch in range(1, args.epochs + 1):
        decoder.model.decoder.train()
        decoder.model.ctc.train()
        random_order.shuffle(examples)
        total_loss = 0.0
        for start in range(0, len(examples), args.accumulate):
            group = examples[start:start + args.accumulate]
            optimizer.zero_grad(set_to_none=True)
            for sample in group:
                encoded = torch.load(sample.encoding, map_location=decoder.device, weights_only=True)
                loss = adaptation_loss(decoder.model, encoded, sample.tokens)
                (loss / len(group)).backward()
                total_loss += loss.item()
            # Stop before applying a nonfinite gradient.
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
        validation = evaluate(decoder, cached["validation"], training_vocabulary)
        overall = validation["overall"]
        key = (overall["wer"], overall["cer"])
        history.append({"epoch": epoch, "loss": total_loss / len(examples), "validation": validation})
        print(f"Epoch {epoch}/{args.epochs}, validation WER {key[0]:.1%}, CER {key[1]:.1%}", flush=True)
        if best is None or key < best[0]:
            best = (key, epoch, validation, adapter_weights(adapters))

    _, best_epoch, best_validation, best_adapter = best
    for name, adapter in adapters.items():
        parent, _, child = name.rpartition(".")
        setattr(decoder.model.get_submodule(parent), child, adapter.layer)
    baseline_test = evaluate(decoder, cached["test"], training_vocabulary)
    merge_weights(decoder.model, best_adapter)
    best_test = evaluate(decoder, cached["test"], training_vocabulary)
    eligible = (
        best_validation["overall"]["wer"] < baseline_validation["overall"]["wer"]
        and best_test["overall"]["wer"] < baseline_test["overall"]["wer"]
        and best_test["overall"]["cer"] <= baseline_test["overall"]["cer"]
    )
    return {
        "base_validation": baseline_validation, "base_test": baseline_test,
        "candidate_validation": best_validation, "candidate_test": best_test,
        "best_epoch": best_epoch, "history": history, "passes_comparison": eligible,
    }, best_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="UTF-8 JSONL with clip, text, speaker and split")
    parser.add_argument("run", type=Path, help="New directory for the report and any passing adapter")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--source", type=Path, default=VISPER_SOURCE)
    parser.add_argument("--task", type=Path, default=ASSETS / "face_landmarker.task")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.rank, args.accumulate) < 1 or args.rank > 64:
        parser.error("epochs and accumulate must be positive and rank must be between 1 and 64")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    try:
        splits = load_manifest(args.manifest)
        args.run.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(args.seed)
        decoder = VisperDecoder(args.source, args.checkpoint, args.device)
        base_sha256 = file_sha256(args.checkpoint)
        print(f"Visual-only training on {decoder.device}", flush=True)
        with tempfile.TemporaryDirectory(prefix="features-", dir=args.run) as cache:
            cached = encode_dataset(decoder, splits, args.source, args.task, Path(cache))
            report, weights = train(decoder, cached, args)
        report["base_sha256"] = base_sha256
        report["manifest_sha256"] = file_sha256(args.manifest)
        report["speakers"] = {split: len({clip.speaker for clip in clips}) for split, clips in splits.items()}
        report["parameters"] = {name: getattr(args, name) for name in ("rank", "learning_rate", "seed", "accumulate")}
        if report["passes_comparison"]:
            torch.save({"version": 1, "base_sha256": base_sha256, "weights": weights}, args.run / "adapter.pt")
        (args.run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        base, candidate = report["base_test"]["overall"], report["candidate_test"]["overall"]
        print(f"Held-out test WER {base['wer']:.1%} base, {candidate['wer']:.1%} adapted")
        print(f"Report written to {args.run / 'report.json'}")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"ViSpeR training failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
