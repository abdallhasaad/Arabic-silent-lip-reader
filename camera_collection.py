"""Opt-in camera capture of mouth crops paired with prompts for training."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np

from visper_training_data import training_text


class CameraCollection:
    def __init__(self, directory: Path, prompts: Path, speaker: str, split: str):
        self.directory = directory.resolve()
        self.manifest = self.directory / "manifest.jsonl"
        self.speaker = speaker.strip().casefold()
        self.split = split
        lines = [line.strip() for line in prompts.read_text(encoding="utf-8-sig").splitlines()]
        if not self.speaker or not any(lines):
            raise ValueError("Provide a speaker code and at least one prompt")
        previous: Counter[str] = Counter()
        if self.manifest.is_file():
            for line in self.manifest.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["speaker"].strip().casefold() != self.speaker:
                    continue
                if record["split"] != self.split:
                    raise ValueError("This speaker already belongs to another split")
                if (self.directory / record["clip"]).is_file():
                    previous[training_text(record["text"])] += 1
        self.prompts = []
        for line in filter(None, lines):
            text = training_text(line)
            if not text:
                raise ValueError("Each prompt must contain words")
            if previous[text]:
                previous[text] -= 1
            else:
                self.prompts.append(line)
        self.completed = 0

    @property
    def prompt(self) -> str | None:
        return self.prompts[self.completed] if self.completed < len(self.prompts) else None

    def save(self, crops: np.ndarray) -> None:
        text = self.prompt
        if text is None:
            raise RuntimeError("All prompts have been recorded")
        folder = self.directory / "mouth"
        folder.mkdir(parents=True, exist_ok=True)
        clip = folder / f"{uuid4().hex}.npy"
        record = {"clip": clip.relative_to(self.directory).as_posix(), "text": text,
                  "speaker": self.speaker, "split": self.split}
        with clip.open("xb") as stream:
            np.save(stream, crops, allow_pickle=False)
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(self.manifest, os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_BINARY, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
        self.completed += 1
