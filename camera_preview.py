"""Read a word or short Arabic phrase from the live webcam."""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import simpledialog
from typing import NamedTuple
from uuid import uuid4

import cv2
import numpy as np
from torch import Tensor

from camera_evaluation import CameraEvaluation, correction_summary
from camera_personalization import CameraPersonalization, LearningProgress, SavedFeedbackError
from camera_preprocessor import (
    LIP_IDS,
    MIN_FACE_DETECTION_RATE,
    VISPER_ALIGNMENT_IDS,
    create_face_landmarker,
    draw_lip_points,
    extract_face_points,
)
from visper_inference import (
    ASSETS, VISPER_SOURCE, VisperDecoder, default_checkpoint, mouth_images, mouth_reference,
    resample_points, spoken_frame_span,
)


SAMPLE_PERIOD_NS = 1_000_000_000 // 25
BASELINE_DURATION_NS = 3_000_000_000
MAX_SPEECH_NS = 5_000_000_000
MIN_WINDOW_NS = 2_000_000_000
MIN_CAPTURE_FPS = 12
MAX_FRAME_GAP_NS = 160_000_000
MAX_EDGE_GAP_NS = 80_000_000
PREPARATION_TIME_NS = 850_000_000
REST_SAMPLE_NS = 480_000_000
PREVIEW_SIZE = (640, 480)
SPEECH_CUES = {
    "idle": ("جاهز للنطق", "#263c4a", "#e6f1f5"),
    "prepare": ("استعد", "#4b351d", "#ffd484"),
    "speak": ("انطق الآن", "#174237", "#b9f5d8"),
    "stop": ("توقّف عن النطق", "#4e2930", "#ffd2d4"),
    "offline": ("الكاميرا غير جاهزة", "#263542", "#e6f1f5"),
}


class LandmarkSample(NamedTuple):
    timestamp_ns: int
    lips: np.ndarray | None
    alignment_points: np.ndarray | None
    gray_frame: np.ndarray


class FeedbackAttempt(NamedTuple):
    id: str
    text: str
    encoded: Tensor
    evaluation_index: int | None = None


class FeedbackReceipt(NamedTuple):
    progress: LearningProgress
    corrected: bool
    comparison: str = ""


class TimedLandmarkWindow:
    def __init__(self, duration_ns: int):
        self.duration_ns = duration_ns
        self.samples: list[LandmarkSample] = []

    def add(
        self, timestamp_ns: int, lips: np.ndarray | None,
        alignment_points: np.ndarray | None, gray_frame: np.ndarray,
    ) -> bool:
        self.samples.append(LandmarkSample(timestamp_ns, lips, alignment_points, gray_frame))
        return self.elapsed_ns >= self.duration_ns

    @property
    def elapsed_ns(self) -> int:
        return self.samples[-1].timestamp_ns - self.samples[0].timestamp_ns if self.samples else 0

    def has_camera_coverage(self) -> bool:
        minimum_frames = self.duration_ns * MIN_CAPTURE_FPS // 1_000_000_000 + 1
        if len(self.samples) < minimum_frames or self.elapsed_ns < self.duration_ns:
            return False
        times = [sample.timestamp_ns for sample in self.samples]
        return bool(np.all(np.diff(times) <= MAX_FRAME_GAP_NS))

    def valid_landmarks(self) -> list[LandmarkSample]:
        valid = [
            sample for sample in self.samples
            if sample.lips is not None and sample.alignment_points is not None
        ]
        if len(valid) < 2 or len(valid) < MIN_FACE_DETECTION_RATE * len(self.samples):
            return []

        start_ns = self.samples[0].timestamp_ns
        times = np.array([sample.timestamp_ns - start_ns for sample in valid], dtype=np.int64)
        if (
            times[0] > MAX_EDGE_GAP_NS
            or times[-1] < self.duration_ns - MAX_EDGE_GAP_NS
            or np.any(np.diff(times) > MAX_FRAME_GAP_NS)
        ):
            return []
        return valid


def open_camera(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LiveCameraPreview:
    def __init__(self, root: tk.Tk, capture, landmarker, transcriber: VisperDecoder | None, *,
                 collection=None, reference=None, personalization: CameraPersonalization | None = None):
        self.root = root
        self.capture = capture
        self.landmarker = landmarker
        self.transcriber = transcriber
        self.collection = collection
        self.personalization = personalization
        self.feedback_attempt: FeedbackAttempt | None = None
        self.submitted_feedback: FeedbackAttempt | None = None
        self.training_request = None
        self.session_confirmed = self.session_corrected = 0
        self.saved_confirmed = self.saved_corrected = 0
        self.reference = reference if collection is not None else transcriber.reference
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.transcript_request = None
        self.window: TimedLandmarkWindow | None = None
        self.evaluation: CameraEvaluation | None = None
        self.camera_ready = False
        self.speak_at = 0
        self.speech_duration_ns = 0
        self.speaking_prompt = False
        self.finish_requested = False
        self.cue_state = None
        self.next_detection_at = 0
        self.next_preview_at = 0
        self.lips = None
        self.preview_image = None

        root.title("قارئ الشفاه العربي")
        root.configure(bg="#101820")
        root.resizable(False, False)

        content = tk.Frame(root, bg="#101820", padx=14, pady=14)
        content.pack()
        content.columnconfigure(1, minsize=336)
        self.canvas = tk.Canvas(
            content, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
            bg="#070d11", highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, padx=(0, 14), sticky="n")
        self.camera_image = self.canvas.create_image(
            PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2,
        )

        panel = tk.Frame(content, bg="#1b2933")
        panel.grid(row=0, column=1, sticky="nsew")
        self.status = tk.StringVar(value="جارٍ فتح الكاميرا…")
        self.face_status = tk.StringVar()

        tk.Label(
            panel, text="جمع عينات الشفاه" if collection else "الكاميرا المباشرة",
            bg="#1b2933", fg="#f0f7f8",
            font=("Segoe UI", 16, "bold"), anchor="e",
        ).pack(fill="x", padx=16, pady=(14, 8))
        self.cue_card = tk.Frame(panel, padx=10, pady=7)
        self.cue_card.pack(fill="x", padx=16)
        self.cue_label = tk.Label(self.cue_card, font=("Segoe UI", 18, "bold"))
        self.cue_label.pack(fill="x")
        self.cue_detail = tk.Label(self.cue_card, font=("Segoe UI", 10))
        self.cue_detail.pack(fill="x")
        self.show_speech_cue("offline", "جارٍ فتح الكاميرا")
        tk.Label(
            panel, textvariable=self.status, bg="#1b2933", fg="#b9dde0",
            font=("Segoe UI", 10), anchor="e", justify="right", wraplength=300,
        ).pack(fill="x", padx=16, pady=(5, 0))
        self.start_button = tk.Button(
            panel, text="ابدأ المحاولة", command=self.start, state="disabled",
            bg="#85e4d5", fg="#102627", activebackground="#b1f2e8", activeforeground="#102627",
            disabledforeground="#486564", font=("Segoe UI", 11, "bold"), relief="flat", pady=4,
        )
        self.start_button.pack(fill="x", padx=16, pady=(7, 5))
        tk.Label(
            panel, textvariable=self.face_status, bg="#1b2933", fg="#abc0ca",
            font=("Segoe UI", 10), anchor="e",
        ).pack(fill="x", padx=16)

        tk.Label(
            panel, text="النص المطلوب" if collection else "النص المقروء", bg="#1b2933", fg="#f0f7f8",
            font=("Segoe UI", 12, "bold"), anchor="e",
        ).pack(fill="x", padx=16, pady=(8, 2))
        self.transcript = tk.StringVar()
        tk.Label(
            panel, textvariable=self.transcript, bg="#243640", fg="#e5f7f7",
            font=("Segoe UI", 15), anchor="ne", justify="right",
            wraplength=283, padx=9, pady=7,
        ).pack(fill="x", padx=16)
        if personalization is not None:
            actions = tk.Frame(panel, bg="#1b2933")
            actions.pack(fill="x", padx=16, pady=(8, 4))
            self.correct_button = tk.Button(
                actions, text="تأكيد النتيجة", command=lambda: self.submit_feedback(False), state="disabled",
                bg="#275747", fg="#e6fff3", activebackground="#376f5a", activeforeground="#ffffff",
                disabledforeground="#9ec0b8", font=("Segoe UI", 10, "bold"), pady=5, relief="flat",
            )
            self.correct_button.pack(side="right", expand=True, fill="x", padx=(3, 0))
            self.fix_button = tk.Button(
                actions, text="تصحيح النتيجة", command=lambda: self.submit_feedback(True), state="disabled",
                bg="#4b395a", fg="#f7e9ff", activebackground="#654d77", activeforeground="#ffffff",
                disabledforeground="#c4b4cf", font=("Segoe UI", 10, "bold"), pady=5, relief="flat",
            )
            self.fix_button.pack(side="right", expand=True, fill="x", padx=(0, 3))
            self.feedback_status = tk.StringVar()
            self.feedback_counts = tk.StringVar()
            self.update_feedback_counts()
            tk.Label(
                panel, textvariable=self.feedback_status, bg="#1b2933", fg="#d5f1e8",
                font=("Segoe UI", 10), anchor="ne", justify="right", wraplength=300,
            ).pack(fill="x", padx=16)
            tk.Label(
                panel, textvariable=self.feedback_counts, bg="#1b2933", fg="#abc0ca",
                font=("Segoe UI", 10), anchor="e", justify="right",
            ).pack(fill="x", padx=16, pady=(2, 0))
            if personalization.database.is_file():
                self.feedback_status.set("جارٍ استعادة التدريب الشخصي…")
                self.training_request = self.executor.submit(personalization.resume)
        self.evaluation_text = tk.StringVar()
        self.evaluation_title = "اختبر أثر التدريب" if personalization else "اختبر دقة الالتقاط"
        self.evaluation_button = tk.Button(
            panel, text=self.evaluation_title, command=self.toggle_evaluation, bg="#294e63", fg="#f1f9ff",
            activebackground="#396b82", activeforeground="#ffffff", disabledforeground="#a5c0cc",
            font=("Segoe UI", 11, "bold"), pady=6, relief="flat",
        )
        if collection is None:
            self.evaluation_button.pack(fill="x", padx=16, pady=(9, 3))
        else:
            self.transcript.set(collection.prompt or "")
            self.evaluation_text.set("تُحفظ منطقة الفم والنص محليًا عند إتمام كل عينة.")
        tk.Label(
            panel, textvariable=self.evaluation_text, bg="#1b2933", fg="#b9dde0",
            font=("Segoe UI", 10), anchor="e", justify="right", wraplength=300,
        ).pack(fill="x", padx=16, pady=(0, 10))

        root.bind("<space>", self.start)
        root.bind("<Escape>", self.close)
        root.bind("<KeyPress-q>", self.close)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def show_speech_cue(self, phase: str, detail: str = ""):
        if phase != self.cue_state:
            title, background, foreground = SPEECH_CUES[phase]
            self.cue_card.configure(bg=background)
            self.cue_label.configure(text=title, bg=background, fg=foreground)
            self.cue_detail.configure(bg=background, fg=foreground)
            self.cue_state = phase
        if self.cue_detail.cget("text") != detail:
            self.cue_detail.configure(text=detail)

    def toggle_evaluation(self):
        if self.window is not None or self.transcript_request is not None or self.training_request is not None:
            return
        self.evaluation = CameraEvaluation(personal=self.personalization is not None) if self.evaluation is None else None
        self.feedback_attempt = None
        self.update_feedback_buttons()
        self.update_evaluation()
        self.transcript.set("")

    def update_evaluation(self):
        evaluation = self.evaluation
        if evaluation is None:
            self.evaluation_button.configure(text=self.evaluation_title)
            self.evaluation_text.set("")
            return
        active = evaluation.prompt is not None
        self.evaluation_button.configure(text="إنهاء الاختبار" if active else "إغلاق نتيجة الاختبار")
        prompt = f"اضغط ابدأ ثم انطق «{evaluation.prompt}»" if active else "اكتمل الاختبار"
        self.evaluation_text.set(f"{prompt}\n{evaluation.summary}".rstrip())

    def start(self, event=None):
        if not self.camera_ready or self.transcript_request is not None or self.training_request is not None:
            return "break"
        if self.collection is not None and self.collection.prompt is None:
            return "break"
        evaluating = self.evaluation is not None and self.evaluation.prompt is not None
        if self.window is not None:
            if self.speaking_prompt and not evaluating:
                self.finish_requested = True
                self.start_button.configure(state="disabled")
                self.show_speech_cue("stop", "جارٍ إنهاء الالتقاط")
                self.status.set("")
            return "break"

        self.speech_duration_ns = BASELINE_DURATION_NS if evaluating and not self.evaluation.personal else MAX_SPEECH_NS
        self.feedback_attempt = None
        self.update_feedback_buttons()
        self.window = TimedLandmarkWindow(self.speech_duration_ns + REST_SAMPLE_NS)
        self.speak_at = time.monotonic_ns() + PREPARATION_TIME_NS
        self.speaking_prompt = self.finish_requested = False
        self.start_button.configure(text="ابدأ المحاولة", state="disabled")
        self.evaluation_button.configure(state="disabled")
        self.show_speech_cue("prepare", "أبقِ شفتيك مرتاحتين")
        self.status.set("")
        if self.collection is None:
            self.transcript.set("")
        return "break"

    def close(self, event=None):
        self.root.destroy()

    def run(self):
        self.root.after(0, self.update)
        try:
            self.root.mainloop()
        finally:
            self.executor.shutdown(wait=False)

    def ready_button(self):
        available = (
            self.camera_ready and self.training_request is None
            and (self.collection is None or self.collection.prompt is not None)
        )
        self.start_button.configure(text="ابدأ المحاولة", state="normal" if available else "disabled")
        self.evaluation_button.configure(state="normal" if self.training_request is None else "disabled")

    def update_feedback_buttons(self):
        if self.personalization is None:
            return
        attempt = self.feedback_attempt
        available = attempt is not None and self.training_request is None
        self.fix_button.configure(state="normal" if available else "disabled")
        self.correct_button.configure(state="normal" if available and attempt.text else "disabled")

    def submit_feedback(self, corrected: bool):
        attempt = self.feedback_attempt
        if attempt is None or self.training_request is not None or (not corrected and not attempt.text):
            return
        text = attempt.text
        evaluation = self.evaluation
        index = attempt.evaluation_index
        if corrected:
            initial = evaluation.observations[index].reference if evaluation is not None and index is not None else text
            text = simpledialog.askstring(
                "تصحيح التنبؤ", "اكتب الكلمة أو العبارة كما نطقتها.",
                initialvalue=initial, parent=self.root,
            )
            if text is None:
                return
        if evaluation is not None and index is not None:
            evaluation.revise_reference(index, text)
            self.update_evaluation()
        self.submitted_feedback = attempt
        self.feedback_status.set("جارٍ حفظ التصحيح والتدريب…" if corrected else "جارٍ حفظ التأكيد والتدريب…")
        self.training_request = self.executor.submit(
            self.train_on_feedback, attempt, text, corrected,
        )
        self.start_button.configure(state="disabled")
        self.evaluation_button.configure(state="disabled")
        self.update_feedback_buttons()

    def train_on_feedback(self, attempt: FeedbackAttempt, text: str, corrected: bool) -> FeedbackReceipt:
        progress = self.personalization.save_feedback(attempt.id, attempt.encoded, text, corrected)
        if not corrected:
            return FeedbackReceipt(progress, corrected)
        try:
            prediction = attempt.text
            if progress.updated or progress.applied:
                prediction = self.transcriber.transcribe_encoded(attempt.encoded.to(self.transcriber.device))
            comparison = correction_summary(text, attempt.text, prediction)
        except (RuntimeError, ValueError) as error:
            print(f"Post-correction comparison failed: {error}", file=sys.stderr)
            comparison = "تعذّرت إعادة قراءة هذه المحاولة."
        return FeedbackReceipt(progress, corrected, comparison)

    def update_feedback_counts(self, progress: LearningProgress | None = None, corrected: bool | None = None):
        if corrected is True:
            self.session_corrected += 1
        elif corrected is False:
            self.session_confirmed += 1
        if progress is not None:
            self.saved_confirmed, self.saved_corrected = progress.confirmed, progress.corrected
        elif corrected is True:
            self.saved_corrected += 1
        elif corrected is False:
            self.saved_confirmed += 1
        self.feedback_counts.set(
            f"هذه الجلسة   {self.session_confirmed} تأكيد · {self.session_corrected} تصحيح\n"
            f"الإجمالي المحفوظ   {self.saved_confirmed} تأكيد · {self.saved_corrected} تصحيح"
        )

    def check_training(self):
        if self.training_request is None or not self.training_request.done():
            return
        committed = False
        try:
            result = self.training_request.result()
            receipt = result if isinstance(result, FeedbackReceipt) else None
            progress = receipt.progress if receipt else result
            committed = True
            self.update_feedback_counts(progress, receipt.corrected if receipt else None)
            if receipt:
                message = "حُفظ التصحيح." if receipt.corrected else "تم تأكيد النتيجة وحفظها."
                if progress.applied is True:
                    message += "\nطُبّق تحديث الأوزان. المحاولة التالية ستستخدمه تلقائيًا."
                elif progress.applied is False:
                    message += "\nلم يجتز التحديث فحص التدريب. ستُستخدم الأوزان المعتمدة سابقًا."
                if receipt.comparison:
                    message += f"\n{receipt.comparison}"
                self.feedback_status.set(message)
            else:
                self.feedback_status.set("التدريب الشخصي جاهز.")
        except SavedFeedbackError as error:
            committed = True
            print(f"Personal training interrupted after saving feedback: {error}", file=sys.stderr)
            self.update_feedback_counts(corrected=error.corrected)
            self.feedback_status.set("حُفظت العينة. سيُستأنف تدريبها عند فتح التطبيق.")
        except ValueError as error:
            print(f"Invalid camera feedback: {error}", file=sys.stderr)
            self.feedback_status.set("أدخل نصًا واضحًا وطوله مناسب للمحاولة.")
        except Exception as error:
            print(f"Personal feedback failed: {error}", file=sys.stderr)
            self.feedback_status.set("تعذّر تشغيل التدريب الشخصي. راجع رسالة الطرفية.")
        finally:
            if committed and self.feedback_attempt is self.submitted_feedback:
                self.feedback_attempt = None
            self.training_request = self.submitted_feedback = None
            self.update_feedback_buttons()
            if self.window is None and self.transcript_request is None:
                self.ready_button()

    def update(self):
        self.check_training()
        if self.transcript_request is not None and self.transcript_request.done():
            try:
                text, previous, encoded = self.transcript_request.result()
                self.transcript.set(text)
                self.status.set("انتهت المحاولة." if text else "لم يُقرأ نص واضح في هذه المحاولة.")
                evaluation_index = None
                if previous is not None and self.evaluation is not None:
                    evaluation_index = self.evaluation.completed
                    self.evaluation.record(previous, text)
                    self.update_evaluation()
                if encoded is not None:
                    self.feedback_attempt = FeedbackAttempt(uuid4().hex, text, encoded, evaluation_index)
                    self.update_feedback_buttons()
            except (RuntimeError, ValueError) as error:
                print(f"ViSpeR inference failed: {error}", file=sys.stderr)
                self.status.set("تعذّر تشغيل قراءة النص. راجع رسالة الخطأ في الطرفية.")
            finally:
                self.transcript_request = None
                self.ready_button()

        ok, frame = self.capture.read()
        now = time.monotonic_ns()
        if not ok or frame is None:
            self.camera_ready = False
            self.window = None
            self.lips = None
            self.status.set("تعذّرت قراءة الكاميرا. تأكد من اتصالها.")
            self.face_status.set("")
            self.show_speech_cue("offline")
            self.ready_button()
            self.root.after(250, self.update)
            return

        if not self.camera_ready:
            self.camera_ready = True
            self.show_speech_cue("idle", "ابدأ عند الاستعداد")
            if self.transcript_request is None:
                complete = self.collection is not None and self.collection.prompt is None
                self.status.set("اكتملت العينات." if complete else "اضغط ابدأ لنطق كلمة أو عبارة.")
                self.ready_button()

        window = self.window
        capturing = window is not None and now >= self.speak_at - REST_SAMPLE_NS
        alignment_points = None
        if capturing or now >= self.next_detection_at:
            ids = LIP_IDS + VISPER_ALIGNMENT_IDS if capturing else LIP_IDS
            points = extract_face_points(frame, self.landmarker, ids)
            self.lips = points[:len(LIP_IDS)] if points is not None else None
            if capturing and points is not None:
                alignment_points = points[len(LIP_IDS):]
            self.face_status.set("الوجه ظاهر" if self.lips is not None else "ضع وجهك أمام الكاميرا")
            self.next_detection_at = now + 75_000_000

        if capturing:
            deadline = self.speak_at + self.speech_duration_ns
            if self.speak_at <= now < deadline and not self.finish_requested:
                if not self.speaking_prompt:
                    self.speaking_prompt = True
                    if self.evaluation is None or self.evaluation.prompt is None:
                        self.start_button.configure(text="انتهيت من النطق", state="normal")
                remaining = (deadline - now) / 1_000_000_000
                self.show_speech_cue("speak", f"الوقت المتبقي {remaining:.1f} ث")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            complete = window.add(now, self.lips, alignment_points, gray)
            if complete or now >= deadline or (self.finish_requested and window.elapsed_ns >= MIN_WINDOW_NS):
                window.duration_ns = min(window.duration_ns, window.elapsed_ns)
                self.finish(window)

        if now >= self.next_preview_at:
            self.show_frame(frame)
            self.next_preview_at = now + (50_000_000 if capturing else 28_000_000)
        self.root.after(1, self.update)

    def finish(self, window: TimedLandmarkWindow):
        self.window = None
        self.show_speech_cue("stop", "انتهى الالتقاط")
        self.ready_button()
        if window.elapsed_ns < MIN_WINDOW_NS or not window.has_camera_coverage():
            self.status.set("تقطّعت إطارات الكاميرا. أعد المحاولة أو خفّض الدقة.")
            return

        valid = window.valid_landmarks()
        if not valid:
            self.status.set("تعذّر تتبّع الشفاه. واجه الكاميرا وأعد المحاولة.")
            return

        start = window.samples[0].timestamp_ns
        times = np.array([sample.timestamp_ns - start for sample in window.samples], dtype=np.int64)
        valid_times = np.array([sample.timestamp_ns - start for sample in valid], dtype=np.int64)
        targets = np.arange(0, window.duration_ns + 1, SAMPLE_PERIOD_NS, dtype=np.int64)
        cue = self.speak_at - start
        lips = resample_points(valid_times, np.stack([sample.lips for sample in valid]), targets)
        eyes = resample_points(
            valid_times, np.stack([sample.alignment_points[:2] for sample in valid]), targets,
        )
        span = spoken_frame_span(lips, eyes, int(np.searchsorted(targets, cue)))
        evaluating = self.evaluation is not None and self.evaluation.prompt is not None
        evaluate_personal = evaluating and self.evaluation.personal
        evaluate_timing = evaluating and not evaluate_personal
        if span is None and not evaluate_timing:
            self.status.set("لم تظهر حركة نطق واضحة. انتظر الإشارة ثم أعد المحاولة.")
            return

        frames = [sample.gray_frame for sample in window.samples]
        anchors = [sample.alignment_points for sample in window.samples]
        crops = None
        if span is not None:
            crops = mouth_images(frames, times, anchors, targets, self.reference)
            if crops is not None:
                crops = crops[span].copy()
        baseline = None
        if evaluate_timing:
            baseline_start = times[min(np.searchsorted(times, cue), len(times) - 1)]
            baseline_targets = np.arange(
                baseline_start, baseline_start + BASELINE_DURATION_NS + 1, SAMPLE_PERIOD_NS,
                dtype=np.int64,
            )
            baseline = mouth_images(frames, times, anchors, baseline_targets, self.reference)
        if (span is not None and crops is None) or (evaluate_timing and baseline is None):
            self.status.set("تعذّرت محاذاة الفم. واجه الكاميرا وأعد المحاولة.")
            return

        if self.collection is not None:
            try:
                self.collection.save(crops)
            except (OSError, ValueError) as error:
                print(f"Could not save camera sample: {error}", file=sys.stderr)
                self.status.set("تعذّر حفظ العينة. راجع رسالة الخطأ في الطرفية.")
                return
            prompt = self.collection.prompt
            self.transcript.set(prompt or "")
            self.status.set("حُفظت العينة. اضغط ابدأ للنص التالي." if prompt else "اكتملت العينات.")
            self.ready_button()
            return

        self.start_button.configure(text="ابدأ المحاولة", state="disabled")
        self.evaluation_button.configure(state="disabled")
        self.status.set("جارٍ قراءة حركة الشفاه…")
        self.transcript_request = self.executor.submit(self.decode_attempt, crops, baseline, evaluate_personal)

    def decode_attempt(self, crops, baseline, evaluate_personal: bool = False):
        if self.personalization is not None:
            try:
                self.personalization.refresh_before_prediction()
            except Exception as error:
                print(f"Could not restore personal weights: {error}", file=sys.stderr)
            if baseline is None and crops is not None:
                encoded = self.transcriber.encode(crops)
                if evaluate_personal:
                    personal, unadapted = self.personalization.compare_predictions(encoded)
                    return personal, unadapted, encoded.cpu()
                text = self.transcriber.transcribe_encoded(encoded)
                return text, None, encoded.cpu()
        text = self.transcriber.transcribe(crops) if crops is not None else ""
        previous = None
        if baseline is not None:
            previous = self.transcriber.transcribe(baseline)
        return text, previous, None

    def show_frame(self, frame: np.ndarray):
        frame = draw_lip_points(frame, self.lips)
        height, width = frame.shape[:2]
        scale = min(PREVIEW_SIZE[0] / width, PREVIEW_SIZE[1] / height)
        if scale != 1.0:
            frame = cv2.resize(
                frame, (max(1, round(width * scale)), max(1, round(height * scale))),
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
        self.preview_image = tk.PhotoImage(master=self.root, data=ppm, format="PPM")
        self.canvas.itemconfigure(self.camera_image, image=self.preview_image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read Arabic speech from a live camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--visper", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check-visper", action="store_true", help="Check inference without opening the camera")
    parser.add_argument("--visper-repo", type=Path, default=VISPER_SOURCE)
    parser.add_argument("--visper-checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--visper-adapter", type=Path, help="Try a locally trained visual adapter")
    parser.add_argument("--feedback-db", type=Path, default=ASSETS.parent / "camera_feedback.sqlite3",
                        help="Local file for personal feedback and weights")
    parser.add_argument("--no-personalization", action="store_true", help="Run without personal feedback or weights")
    parser.add_argument("--collect", type=Path, help="Collect prompted mouth crops in this local directory")
    parser.add_argument("--prompts", type=Path, help="UTF-8 file with one utterance per line")
    parser.add_argument("--speaker", help="Anonymous speaker code for collection")
    parser.add_argument("--split", choices=("train", "validation", "test"), help="Dataset split for collection")
    parser.add_argument(
        "--task", type=Path, default=ASSETS / "face_landmarker.task",
        help="Path to the MediaPipe Face Landmarker model",
    )
    args = parser.parse_args()
    if args.camera < 0 or args.width < 1 or args.height < 1:
        parser.error("camera must be non-negative, and width and height must be positive")
    if not args.task.is_file():
        parser.error(f"Face Landmarker task file not found: {args.task}")
    if args.collect and (not all((args.prompts, args.speaker, args.split)) or args.visper_adapter or args.check_visper):
        parser.error("--collect requires --prompts, --speaker and --split and cannot run with inference checks or an adapter")
    if not args.collect and any((args.prompts, args.speaker, args.split)):
        parser.error("--prompts, --speaker and --split require --collect")

    transcriber = collection = reference = personalization = None
    try:
        if args.collect:
            from camera_collection import CameraCollection

            collection = CameraCollection(args.collect, args.prompts, args.speaker, args.split)
            if collection.prompt is None:
                print("All prompts have already been collected for this speaker.")
                return 0
            reference = mouth_reference(args.visper_repo)
        else:
            transcriber = VisperDecoder(args.visper_repo, args.visper_checkpoint)
            if args.visper_adapter:
                from visper_adapter import merge_adapter

                merge_adapter(transcriber.model, args.visper_checkpoint, args.visper_adapter)
            if not args.no_personalization and not args.check_visper:
                personalization = CameraPersonalization(
                    transcriber, args.visper_repo, args.visper_checkpoint, args.visper_adapter, args.feedback_db,
                )
        if args.check_visper:
            transcriber.transcribe(np.full((25, 96, 96), 128, dtype=np.uint8))
            print(f"ViSpeR Arabic model passed the inference check on {transcriber.device}.")
            return 0
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Could not initialize camera session: {error}\n")

    with create_face_landmarker(args.task) as landmarker:
        cap = open_camera(args.camera, args.width, args.height)
        try:
            if not cap.isOpened():
                parser.exit(1, f"Could not open camera {args.camera}. Try --camera 1 or check camera access.\n")
            LiveCameraPreview(
                tk.Tk(), cap, landmarker, transcriber, collection=collection,
                reference=reference, personalization=personalization,
            ).run()
        finally:
            cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
