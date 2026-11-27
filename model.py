import torch
import torch.nn as nn


class ArabicLipGeometryNet(nn.Module):

    def __init__(
        self,
        num_classes=100,
        hidden_size=96,
        dropout=0.25
    ):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.temporal_gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, points):

        batch_size, frames, _, _ = points.shape

        absolute = points.reshape(
            batch_size, frames, 80
        )

        delta = torch.zeros_like(points)

        delta[:, 1:] = (
            points[:, 1:] - points[:, :-1]
        )

        delta = delta.reshape(
            batch_size, frames, 80
        )

        features = torch.cat(
            [absolute, delta],
            dim=-1
        )

        features = self.frame_encoder(features)

        temporal_features, _ = self.temporal_gru(
            features
        )

        attention_scores = self.attention(
            temporal_features
        ).squeeze(-1)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        pooled = torch.sum(
            temporal_features *
            attention_weights.unsqueeze(-1),
            dim=1
        )

        return self.classifier(pooled)
