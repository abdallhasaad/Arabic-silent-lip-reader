"""Small residual updates for the visual ViSpeR decoder and CTC head."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adaptable_layers(model):
    return {
        name: layer for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear) and (
            name == "ctc.ctc_lo"
            or (
                name.startswith("decoder.decoders.")
                and name.endswith((".src_attn.linear_q", ".src_attn.linear_v"))
            )
        )
    }


class LowRankLinear(nn.Module):
    def __init__(self, layer: nn.Linear, rank: int):
        super().__init__()
        self.layer = layer
        self.enabled = True
        self.down = nn.Linear(layer.in_features, rank, bias=False, device=layer.weight.device, dtype=layer.weight.dtype)
        self.up = nn.Linear(rank, layer.out_features, bias=False, device=layer.weight.device, dtype=layer.weight.dtype)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.layer(x)
        return self.layer(x) + self.up(self.down(x))


def install_adapters(model, rank: int) -> dict[str, LowRankLinear]:
    model.requires_grad_(False)
    adapters = {}
    targets = adaptable_layers(model)
    if len(targets) != 2 * len(model.decoder.decoders) + 1:
        raise ValueError("Unexpected ViSpeR decoder and CTC architecture")
    for name, layer in targets.items():
        parent, _, child = name.rpartition(".")
        adapters[name] = LowRankLinear(layer, rank)
        setattr(model.get_submodule(parent), child, adapters[name])
    return adapters


def adapter_weights(adapters) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: (layer.down.weight.detach().to("cpu", copy=True), layer.up.weight.detach().to("cpu", copy=True))
        for name, layer in adapters.items()
    }


@torch.no_grad()
def restore_adapter_weights(adapters, weights) -> None:
    if not isinstance(weights, dict) or set(adapters) != set(weights):
        raise ValueError("Saved weights do not match the personal adapter")
    for name, layer in adapters.items():
        down, up = weights[name]
        if (
            not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor)
            or down.shape != layer.down.weight.shape or up.shape != layer.up.weight.shape
            or not torch.isfinite(down).all() or not torch.isfinite(up).all()
        ):
            raise ValueError(f"Invalid saved personal weights for {name}")
    for name, layer in adapters.items():
        down, up = weights[name]
        layer.down.weight.copy_(down)
        layer.up.weight.copy_(up)


@torch.no_grad()
def merge_adapter(model, base: Path, adapter: Path) -> None:
    payload = torch.load(adapter, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict) or payload.get("version") != 1
        or payload.get("base_sha256") != file_sha256(base)
    ):
        raise ValueError("The adapter is incompatible with this ViSpeR checkpoint")
    merge_weights(model, payload.get("weights"))


@torch.no_grad()
def merge_weights(model, weights) -> None:
    targets = adaptable_layers(model)
    if not isinstance(weights, dict) or set(weights) != set(targets):
        raise ValueError("Adapter layers do not match this ViSpeR model")
    for name, layer in targets.items():
        pair = weights[name]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"Invalid adapter tensors for {name}")
        down, up = pair
        if (
            not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor)
            or not down.is_floating_point() or not up.is_floating_point()
            or down.ndim != 2 or up.ndim != 2
            or down.shape[1] != layer.in_features
            or up.shape != (layer.out_features, down.shape[0])
            or not 1 <= down.shape[0] <= 64
            or not torch.isfinite(down).all() or not torch.isfinite(up).all()
        ):
            raise ValueError(f"Invalid adapter tensors for {name}")
    for name, layer in targets.items():
        down, up = weights[name]
        update = up.to(layer.weight) @ down.to(layer.weight)
        # Replace the mapped parameter so the original checkpoint stays read-only.
        layer.weight = nn.Parameter(layer.weight.detach() + update, requires_grad=False)
