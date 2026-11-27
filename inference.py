from pathlib import Path
import json

import numpy as np
import torch

from model import ArabicLipGeometryNet


class GeometryPredictor:

    def __init__(self, bundle_dir=None, device=None):

        self.bundle_dir = (
            Path(bundle_dir)
            if bundle_dir is not None
            else Path(__file__).resolve().parent
        )

        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        )

        with open(
            self.bundle_dir / "config.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.config = json.load(f)

        with open(
            self.bundle_dir / "labels.json",
            "r",
            encoding="utf-8"
        ) as f:
            label_data = json.load(f)

        self.labels = label_data["labels"]

        if len(self.labels) != self.config["num_classes"]:
            raise ValueError("Label count does not match model config.")

        normalization = np.load(
            self.bundle_dir / "landmark_normalization.npz"
        )

        self.mean = normalization[
            "landmark_mean"
        ].astype(np.float32)

        self.std = normalization[
            "effective_std"
        ].astype(np.float32)

        if self.mean.shape != (40, 2):
            raise ValueError("Invalid landmark mean shape.")

        if self.std.shape != (40, 2):
            raise ValueError("Invalid landmark std shape.")

        self.model = ArabicLipGeometryNet(
            num_classes=self.config["num_classes"],
            hidden_size=self.config["hidden_size"],
            dropout=self.config["dropout"]
        )

        weights = torch.load(
            self.bundle_dir / "weights.pt",
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(
            weights,
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()

    def prepare_landmarks(self, landmarks):

        points = np.asarray(
            landmarks,
            dtype=np.float32
        )

        expected_shape = (
            self.config["num_frames"],
            self.config["num_points"],
            2
        )

        if points.shape != expected_shape:
            raise ValueError(
                f"Expected landmarks {expected_shape}, "
                f"received {points.shape}"
            )

        if not np.isfinite(points).all():
            raise ValueError(
                "Landmarks contain NaN or infinite values."
            )

        # Input is expected in normalized 96x96 ROI coordinates.
        if np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError("Landmarks must be normalized to [0, 1].")
        normalized = (
            points - self.mean
        ) / self.std

        normalized = np.clip(
            normalized,
            -8.0,
            8.0
        )

        return np.ascontiguousarray(
            normalized,
            dtype=np.float32
        )

    @torch.inference_mode()
    def predict(self, landmarks, top_k=5):

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        points = self.prepare_landmarks(
            landmarks
        )

        input_tensor = torch.from_numpy(
            points
        ).unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)

        probabilities = torch.softmax(
            logits.float(),
            dim=1
        )[0]

        k = min(
            top_k,
            len(self.labels)
        )

        values, indices = torch.topk(
            probabilities,
            k=k
        )

        results = []

        for probability, index in zip(
            values.cpu().tolist(),
            indices.cpu().tolist()
        ):

            results.append({
                "label_id": int(index),
                "word": self.labels[index],
                "probability": float(probability)
            })

        return {
            "prediction": results[0],
            "top_k": results
        }
