"""Camera-side preprocessing for ArabicLipGeometryNet v2.

This module mirrors the landmark preprocessing used by the LRW-AR training
notebook. It does not record or write camera frames to disk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

NUM_FRAMES = 29
CROP_SIZE = 96
ALIGN_CANVAS_SIZE = 256
TARGET_EYE_DISTANCE = 116.0
EYE_TARGET_X = ALIGN_CANVAS_SIZE / 2.0
EYE_TARGET_Y = 80.0
MIN_FACE_DETECTION_RATE = 0.75
MIN_VALID_FRAMES = math.ceil(NUM_FRAMES * MIN_FACE_DETECTION_RATE)

# Exact ordered 40-point selection used during LRW-AR preprocessing.
LIP_IDS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]
EYE_IDS = [33, 263]
# ViSpeR's five alignment points in MediaPipe Face Landmarker order.
VISPER_ALIGNMENT_IDS = [33, 263, 2, 61, 291]

if len(LIP_IDS) != 40:
    raise RuntimeError("LIP_IDS must contain exactly 40 landmarks")


@dataclass
class PreprocessedWindow:
    """One 29-frame result in the exact model input coordinate convention."""

    landmarks: np.ndarray       # float32 [29, 40, 2], coordinates in [0, 1]
    mouth_frames: np.ndarray | None  # uint8 [29, 96, 96] when crops are requested
    detection_rate: float


def create_face_landmarker(task_path: str | Path):
    """Create the same IMAGE-mode MediaPipe Tasks FaceLandmarker used in training."""
    task_path = Path(task_path)
    if not task_path.is_file():
        raise FileNotFoundError(f"Face Landmarker task file not found: {task_path}")
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def extract_face_points(frame_bgr: np.ndarray, landmarker, ids: Sequence[int]) -> np.ndarray | None:
    """Return selected landmarks in the source camera's pixel coordinates."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a BGR image with shape [H, W, 3]")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]
    height, width = frame_bgr.shape[:2]
    points = np.asarray(
        [[face[index].x * width, face[index].y * height] for index in ids],
        dtype=np.float32,
    )
    return points if np.isfinite(points).all() else None


def extract_raw_geometry(frame_bgr: np.ndarray, landmarker) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return raw pixel coordinates for 40 lip points and 2 eye corners."""
    points = extract_face_points(frame_bgr, landmarker, LIP_IDS + EYE_IDS)
    if points is None:
        return None, None
    return points[:len(LIP_IDS)], points[len(LIP_IDS):]


def interpolate_landmark_sequence(points: np.ndarray) -> Optional[np.ndarray]:
    """Match training's frame-wise linear interpolation and 75% detection rule."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (NUM_FRAMES, 42, 2):
        raise ValueError(f"Expected combined landmarks {(NUM_FRAMES, 42, 2)}, got {points.shape}")

    valid = np.where(np.isfinite(points).all(axis=(1, 2)))[0]
    if len(valid) < MIN_VALID_FRAMES:
        return None
    if len(valid) == NUM_FRAMES:
        return points

    output = points.copy()
    frame_ids = np.arange(NUM_FRAMES)
    for point_index in range(points.shape[1]):
        for dimension in range(2):
            output[:, point_index, dimension] = np.interp(
                frame_ids,
                valid,
                points[valid, point_index, dimension],
            )
    return output


def similarity_alignment_matrix(eye_a: np.ndarray, eye_b: np.ndarray) -> Optional[np.ndarray]:
    """Exact two-eye-corner similarity alignment used by the training notebook."""
    left, right = (eye_a, eye_b) if eye_a[0] <= eye_b[0] else (eye_b, eye_a)
    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    distance = math.hypot(dx, dy)
    if distance < 5.0:
        return None

    angle = math.degrees(math.atan2(dy, dx))
    scale = TARGET_EYE_DISTANCE / distance
    source_center = (left + right) / 2.0
    matrix = cv2.getRotationMatrix2D(
        (float(source_center[0]), float(source_center[1])), angle, scale
    ).astype(np.float32)
    current_center = matrix[:, :2] @ source_center + matrix[:, 2]
    desired_center = np.asarray([EYE_TARGET_X, EYE_TARGET_Y], dtype=np.float32)
    matrix[:, 2] += desired_center - current_center
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(np.asarray(points, dtype=np.float32)[None, :, :], matrix)[0]


def preprocess_window(
    frames_bgr: Sequence[np.ndarray] | None,
    lip_sequence: Sequence[Optional[np.ndarray]],
    eye_sequence: Sequence[Optional[np.ndarray]],
) -> Optional[PreprocessedWindow]:
    """Convert a 29-frame camera window to the model's [29,40,2] input.

    Raw lip/eye points are interpolated BEFORE per-frame alignment, matching
    the original training implementation. Pass None for `frames_bgr` when
    only the landmarks are needed.
    """
    if len(lip_sequence) != NUM_FRAMES or len(eye_sequence) != NUM_FRAMES:
        raise ValueError(f"All inputs must contain exactly {NUM_FRAMES} frames")
    if frames_bgr is not None and len(frames_bgr) != NUM_FRAMES:
        raise ValueError(f"All inputs must contain exactly {NUM_FRAMES} frames")

    lip_rows = []
    eye_rows = []
    for lips, eyes in zip(lip_sequence, eye_sequence):
        if lips is None or eyes is None:
            lip_rows.append(np.full((40, 2), np.nan, dtype=np.float32))
            eye_rows.append(np.full((2, 2), np.nan, dtype=np.float32))
        else:
            lips = np.asarray(lips, dtype=np.float32)
            eyes = np.asarray(eyes, dtype=np.float32)
            if lips.shape != (40, 2) or eyes.shape != (2, 2):
                raise ValueError("Each detected frame must have lips [40,2] and eyes [2,2]")
            lip_rows.append(lips)
            eye_rows.append(eyes)

    combined = np.concatenate([np.asarray(lip_rows), np.asarray(eye_rows)], axis=1)
    detection_rate = float(np.isfinite(combined).all(axis=(1, 2)).mean())
    combined = interpolate_landmark_sequence(combined)
    if combined is None:
        return None

    lips_all = combined[:, :40, :]
    eyes_all = combined[:, 40:, :]
    mouth_frames = None
    if frames_bgr is not None:
        mouth_frames = np.empty((NUM_FRAMES, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    mouth_points = np.zeros((NUM_FRAMES, 40, 2), dtype=np.float32)

    for time_index in range(NUM_FRAMES):
        matrix = similarity_alignment_matrix(eyes_all[time_index, 0], eyes_all[time_index, 1])
        if matrix is None:
            return None

        aligned_lips = transform_points(lips_all[time_index], matrix)
        center = aligned_lips.mean(axis=0)
        crop_x0 = int(round(float(center[0]) - CROP_SIZE / 2.0))
        crop_y0 = int(round(float(center[1]) - CROP_SIZE / 2.0))

        crop_origin = np.asarray([crop_x0, crop_y0], dtype=np.float32)
        local_points = aligned_lips - crop_origin
        normalized = local_points / float(CROP_SIZE)
        if np.mean((normalized >= 0.0) & (normalized <= 1.0)) < 0.95:
            return None
        mouth_points[time_index] = np.clip(normalized, 0.0, 1.0)

        if frames_bgr is not None and mouth_frames is not None:
            matrix[:, 2] -= crop_origin
            crop = cv2.warpAffine(
                frames_bgr[time_index],
                matrix,
                (CROP_SIZE, CROP_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            mouth_frames[time_index] = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if not np.isfinite(mouth_points).all():
        return None
    return PreprocessedWindow(
        landmarks=mouth_points,
        mouth_frames=mouth_frames,
        detection_rate=detection_rate,
    )


def draw_lip_points(frame_bgr: np.ndarray, lips: Optional[np.ndarray]) -> np.ndarray:
    """Draw the 40 detected lip points on a copy of the camera frame."""
    output = frame_bgr.copy()
    if lips is not None:
        for x, y in np.asarray(lips):
            cv2.circle(output, (int(round(float(x))), int(round(float(y)))), 2, (0, 255, 0), -1)
    return output
