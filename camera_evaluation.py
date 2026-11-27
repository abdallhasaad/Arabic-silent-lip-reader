"""Score two decodes of the same live camera attempt in memory."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


PROMPTS = (
    "أنا", "أريد", "أكلت", "اليوم",
    "صباح الخير", "أريد أن أشرب الماء", "سأعود بعد قليل", "أين وضعت المفاتيح",
)
PERSONAL_PROMPTS = (
    "الآن", "قريب", "تفضل", "غداً",
    "متى تصل إلى البيت", "الباب على اليمين", "دعنا نلتقي مساء", "وضعت الكتاب على الطاولة",
)
ARABIC_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ـ": None})


def word_tokens(text: str) -> list[str]:
    letters = unicodedata.normalize("NFKC", text).translate(ARABIC_VARIANTS)
    return "".join(
        char if char.isalnum() else " "
        for char in letters
        if not unicodedata.category(char).startswith("M")
    ).split()


def word_edits(reference: list[str], prediction: list[str]) -> int:
    row = list(range(len(prediction) + 1))
    for index, word in enumerate(reference, 1):
        next_row = [index]
        for offset, predicted in enumerate(prediction, 1):
            next_row.append(min(
                row[offset] + 1, next_row[-1] + 1, row[offset - 1] + (word != predicted),
            ))
        row = next_row
    return row[-1]


def correction_summary(reference: str, previous: str, current: str) -> str:
    words = word_tokens(reference)
    before = word_edits(words, word_tokens(previous))
    after = word_edits(words, word_tokens(current))
    if before == after:
        return f"على هذه المحاولة، بقي عدد أخطاء الكلمات {after}."
    return f"على هذه المحاولة، عدد أخطاء الكلمات من {before} إلى {after}."


@dataclass
class EvaluationObservation:
    reference: str
    previous: str
    current: str


@dataclass
class CameraEvaluation:
    personal: bool = False
    observations: list[EvaluationObservation] = field(default_factory=list, init=False)

    @property
    def completed(self) -> int:
        return len(self.observations)

    @property
    def reference_words(self) -> int:
        return sum(len(word_tokens(item.reference)) for item in self.observations)

    @property
    def previous_errors(self) -> int:
        return sum(word_edits(word_tokens(item.reference), word_tokens(item.previous)) for item in self.observations)

    @property
    def current_errors(self) -> int:
        return sum(word_edits(word_tokens(item.reference), word_tokens(item.current)) for item in self.observations)

    @property
    def prompt(self) -> str | None:
        prompts = PERSONAL_PROMPTS if self.personal else PROMPTS
        return prompts[self.completed] if self.completed < len(prompts) else None

    def record(self, previous: str, current: str):
        prompt = self.prompt
        if prompt is not None:
            self.observations.append(EvaluationObservation(prompt, previous, current))

    def revise_reference(self, index: int, spoken_text: str):
        if word_tokens(spoken_text):
            self.observations[index].reference = spoken_text

    @property
    def summary(self) -> str:
        reference_words = self.reference_words
        if reference_words == 0:
            return ""
        before = self.previous_errors / reference_words
        after = self.current_errors / reference_words
        if not self.personal:
            return f"معدل خطأ الكلمات في {self.completed} محاولات\nالسابق {before:.0%}  |  الحالي {after:.0%}"
        if after == before:
            change = "معدل الخطأ متساوٍ"
        elif after < before:
            change = f"انخفض الخطأ {100 * (before - after):.1f} نقطة مئوية"
        else:
            change = f"زاد الخطأ {100 * (after - before):.1f} نقطة مئوية"
        return (
            f"خطأ الكلمات في {self.completed} محاولات جديدة\n"
            f"دون التخصيص ({before:.0%})   |   مع التخصيص ({after:.0%})\n{change}"
        )
