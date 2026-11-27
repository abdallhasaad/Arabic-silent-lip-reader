"""Load the upstream ViSpeR visual model and prepare an in-memory camera clip."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parent
ASSETS = PROJECT / "assets"
VISPER_SOURCE = ASSETS / "visper"
REFERENCE_IDS = [36, 45, 33, 48, 54]
MOUTH_SIZE = 96
MODEL_SIZE = 88
FACE_SIZE = 256
SPEECH_CONTEXT_FRAMES = 50
SPEECH_MARGIN_FRAMES = 6

# Matches conf/model/visual_backbone/resnet_conformer.yaml in the ViSpeR
# source release, with the decoder dimensions resolved to the encoder's.
VISUAL_BACKBONE = SimpleNamespace(
    adim=768, aheads=12, eunits=3072, elayers=12,
    transformer_input_layer="conv3d", dropout_rate=0.1,
    transformer_attn_dropout_rate=0.1, transformer_encoder_attn_layer_type="rel_mha",
    macaron_style=True, use_cnn_module=True, cnn_module_kernel=31,
    zero_triu=False, a_upsample_ratio=1, relu_type="swish",
    ddim=768, dheads=12, dunits=3072, dlayers=6, lsm_weight=0.1,
    transformer_length_normalized_loss=False, mtlalpha=0.1, ctc_type="builtin",
)


def default_checkpoint() -> Path:
    checkpoint = PROJECT / "visper_vsr_base.pth"
    return checkpoint if checkpoint.is_file() else ASSETS / checkpoint.name


def mouth_reference(source: Path) -> np.ndarray:
    mean_face = np.load(source / "data_prepare" / "20words_mean_face.npy", allow_pickle=False)
    reference = np.asarray(mean_face[REFERENCE_IDS], dtype=np.float32)
    if reference.shape != (len(REFERENCE_IDS), 2) or not np.isfinite(reference).all():
        raise ValueError("Invalid ViSpeR face reference.")
    return reference


def resample_points(times, points, targets):
    coordinates = points.reshape(len(points), -1)
    return np.array(
        [np.interp(targets, times, column) for column in coordinates.T], dtype=np.float32,
    ).T.reshape(len(targets), *points.shape[1:])


def spoken_frame_span(lips: np.ndarray, eyes: np.ndarray, cue_index: int) -> slice | None:
    """Locate sustained lip deformation independently of 2D head movement."""
    if len(lips) < 5:
        return None
    axis = eyes[:, 1] - eyes[:, 0]
    distance = np.linalg.norm(axis, axis=1)
    if np.any(distance < 5):
        return None

    unit_axis = axis / distance[:, None]
    vertical = np.column_stack((-unit_axis[:, 1], unit_axis[:, 0]))
    rotation = np.stack((unit_axis, vertical), axis=-1)
    centered = lips - lips.mean(axis=1, keepdims=True)
    shape = (centered @ rotation) / distance[:, None, None]

    # Compare contours 80 ms apart and estimate tracker noise during the cue.
    lag = 2
    motion = np.sqrt(np.mean(np.sum((shape[lag:] - shape[:-lag]) ** 2, axis=2), axis=1))
    baseline = motion[:max(0, cue_index - lag)]
    noise = float(np.median(baseline)) if baseline.size else 0.0
    spread = float(np.median(abs(baseline - noise))) if baseline.size else 0.0
    threshold = max(0.0035, noise + 5.0 * spread)
    moving = motion > threshold
    moving[:max(0, cue_index - lag - 1)] = False
    active = np.flatnonzero(moving & (np.convolve(moving, np.ones(3), mode="same") >= 2))
    if not active.size:
        return None

    start = max(0, int(active[0]) - SPEECH_MARGIN_FRAMES)
    stop = min(len(lips), int(active[-1]) + lag + SPEECH_MARGIN_FRAMES + 1)
    missing_context = max(0, SPEECH_CONTEXT_FRAMES - (stop - start))
    start = max(0, start - missing_context // 2)
    stop = min(len(lips), max(stop, start + SPEECH_CONTEXT_FRAMES))
    start = max(0, min(start, stop - SPEECH_CONTEXT_FRAMES))
    return slice(start, stop)


def mouth_images(frames, times, anchors, targets, reference):
    """Sample the camera timeline and align its mouth crops to ViSpeR's reference."""
    valid = [index for index, points in enumerate(anchors) if points is not None]
    if not valid:
        return None

    # Interpolate landmarks on the capture timeline. Sample the actual camera
    # images independently so interpolating landmarks cannot blend two images.
    points = resample_points(times[valid], np.stack([anchors[index] for index in valid]), targets)
    after = np.minimum(np.searchsorted(times, targets), len(times) - 1)
    before = np.maximum(after - 1, 0)
    sampled = np.where(abs(targets - times[before]) <= abs(times[after] - targets), before, after)

    crops = []
    for index, frame_index in enumerate(sampled):
        radius = min(6, index, len(points) - 1 - index)
        smoothed = points[index - radius:index + radius + 1].mean(axis=0)
        smoothed += points[index].mean(axis=0) - smoothed.mean(axis=0)
        matrix, _ = cv2.estimateAffinePartial2D(smoothed, reference, method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():
            return None

        corners = smoothed[-2:] @ matrix[:, :2].T + matrix[:, 2]
        half_size = MOUTH_SIZE // 2
        center = np.clip(np.rint(corners.mean(axis=0)), half_size, FACE_SIZE - half_size)
        matrix[:, 2] -= center - half_size
        crop = cv2.warpAffine(
            frames[frame_index], matrix, (MOUTH_SIZE, MOUTH_SIZE), borderMode=cv2.BORDER_CONSTANT,
        )
        crops.append(crop)
    return np.stack(crops)


def model_input(crops: np.ndarray) -> torch.Tensor:
    """Apply the upstream visual validation crop and normalization."""
    if crops.ndim != 3 or crops.shape[1:] != (MOUTH_SIZE, MOUTH_SIZE) or not len(crops):
        raise ValueError("Expected a non-empty sequence of 96x96 grayscale mouth crops.")
    offset = (MOUTH_SIZE - MODEL_SIZE) // 2
    center = crops[:, offset:offset + MODEL_SIZE, offset:offset + MODEL_SIZE]
    pixels = torch.from_numpy(np.ascontiguousarray(center)).unsqueeze(1).float()
    return (pixels / 255.0 - 0.421) / 0.165


class VisperDecoder:
    def __init__(self, source: Path, checkpoint: Path, device: str | None = None):
        source = source.resolve()
        dictionary = source / "spm" / "unigram" / "unigram_units.txt"
        if not (source / "espnet" / "nets" / "pytorch_backend" / "e2e_asr_conformer.py").is_file():
            raise FileNotFoundError(f"ViSpeR source checkout missing in {source}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ViSpeR visual-only checkpoint missing at {checkpoint}")

        self.reference = mouth_reference(source)
        units = [line.split()[0] for line in dictionary.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokens = ["<blank>", *units, "<eos>"]
        self.language = self.tokens.index("<ar>")

        # The upstream training runner imports audio and training packages.
        # Its visual encoder and beam search can run directly from PyTorch.
        sys.path.insert(0, str(source))
        try:
            from espnet.nets.batch_beam_search import BatchBeamSearch
            from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
            from espnet.nets.scorers.ctc import CTCPrefixScorer
        finally:
            sys.path.pop(0)

        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        if "state_dict" in state:
            state = {
                name.removeprefix("model."): value
                for name, value in state["state_dict"].items()
                if name.startswith("model.")
            }
        self.model = E2E(len(self.tokens), VISUAL_BACKBONE)
        self.model.load_state_dict(state, strict=True, assign=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.search = BatchBeamSearch(
            beam_size=40,
            vocab_size=len(self.tokens),
            weights={"decoder": 0.9, "ctc": 0.1},
            scorers={
                "decoder": self.model.decoder,
                "ctc": CTCPrefixScorer(self.model.ctc, self.model.eos),
            },
            sos=self.model.sos,
            eos=self.model.eos,
            token_list=self.tokens,
            pre_beam_score_key="decoder",
        ).eval()

    @torch.inference_mode()
    def encode(self, crops: np.ndarray) -> torch.Tensor:
        inputs = model_input(crops).unsqueeze(0).to(self.device)
        encoded, _ = self.model.encoder(inputs, None)
        return encoded.squeeze(0)

    @torch.inference_mode()
    def transcribe(self, crops: np.ndarray) -> str:
        return self.transcribe_encoded(self.encode(crops))

    @torch.inference_mode()
    def transcribe_encoded(self, encoded: torch.Tensor) -> str:
        hypotheses = self.search(encoded, lang=self.language)
        if not hypotheses:
            return ""
        indices = hypotheses[0].yseq[2:].tolist()  # Drop the start and Arabic language tokens.
        text = "".join(self.tokens[index] for index in indices if index != self.model.eos)
        return " ".join(text.replace("▁", " ").replace("<unk>", "").replace("<blank>", "").split())
