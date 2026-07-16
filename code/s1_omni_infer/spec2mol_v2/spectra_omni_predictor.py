"""IR vision hidden -> 20-dim motif predictor.

This module wraps the E1 ``PrefixAttnExtractor`` checkpoint that maps a Qwen3-VL
IR vision-span hidden state (shape ``[B, 546, 5120]``) to 20 motif-class logits.

It intentionally does NOT load the VLM. The caller (``vlm_runtime`` /
``infer_merge``) is responsible for producing the vision-span hidden state from
the merged HF model; this predictor only consumes that hidden state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from spec2mol_v2.prefix_attn_extractor import PrefixAttnExtractor
from spec2mol_v2.vision_span import EXPECTED_VISION_SPAN, IMAGE_PAD_ID, VISION_END_ID, VISION_START_ID, slice_vision_span

# Default checkpoint location (populated by prepare_merge_assets.py).


def load_extractor(
    ckpt_pth: str | Path,
    num_classes: int = 20,
    strict: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Build a ``PrefixAttnExtractor`` and load ``ckpt_pth`` into it."""
    ckpt_pth = Path(ckpt_pth)
    blob = torch.load(ckpt_pth, map_location="cpu", weights_only=False)
    args = blob.get("args") or {}
    extractor = PrefixAttnExtractor(
        in_dim=5120,
        proj_dim=int(args.get("proj_dim", 768)),
        n_layers=int(args.get("n_layers", 2)),
        n_heads=int(args.get("n_heads", 8)),
        ffn_mult=int(args.get("ffn_mult", 2)),
        dropout=float(args.get("dropout", 0.1)),
        num_queries=int(args.get("num_queries", 4)),
        num_classes=int(blob.get("num_classes", num_classes)),
        feat_dim=int(args.get("feat_dim", 0)),
        frozen_head_state=None,
    )
    missing, unexpected = extractor.load_state_dict(blob["state_dict"], strict=strict)
    if missing:
        print(f"[spectra_omni_predictor] missing keys: {missing}")
    if unexpected:
        print(f"[spectra_omni_predictor] unexpected keys: {unexpected}")
    return extractor, blob


def _resolve_thresholds(blob: dict[str, Any], num_classes: int) -> list[float]:
    """thresholds_tuned -> thresholds_default -> 0.5 per class."""
    raw = blob.get("thresholds_tuned") or blob.get("thresholds_default")
    if raw is None:
        return [0.5] * num_classes
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().tolist()
    values = [float(x) for x in raw]
    if len(values) < num_classes:
        values = values + [0.5] * (num_classes - len(values))
    return values[:num_classes]


class SpectraOmniMotifPredictor:
    """IR vision-span hidden -> 20-dim motif probabilities / one-hot."""

    def __init__(
        self,
        extractor_ckpt: str | Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.checkpoint = str(Path(extractor_ckpt).resolve())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor, blob = load_extractor(self.checkpoint)
        self.num_classes = int(blob.get("num_classes", 20))
        self.vocab = list(blob.get("vocab") or [f"motif_{i}" for i in range(self.num_classes)])
        self.thresholds_tuned = blob.get("thresholds_tuned")
        self.thresholds_default = blob.get("thresholds_default")
        self.thresholds = _resolve_thresholds(blob, self.num_classes)
        self.threshold_source = (
            "thresholds_tuned" if blob.get("thresholds_tuned") is not None
            else "thresholds_default" if blob.get("thresholds_default") is not None
            else "0.5"
        )

        target_dtype = dtype if dtype is not None else next(self.extractor.parameters()).dtype
        self.extractor = self.extractor.to(device=self.device, dtype=target_dtype)
        self.extractor.eval()

    @torch.no_grad()
    def predict_from_ir_vision_span(self, vision_span: torch.Tensor) -> dict[str, Any]:
        """Run the extractor on ``vision_span`` ``[B, 546, 5120]`` -> motif dict."""
        if vision_span.dim() != 3:
            raise ValueError(f"expected vision_span [B, 546, 5120], got shape {tuple(vision_span.shape)}")
        if vision_span.size(1) != EXPECTED_VISION_SPAN:
            raise ValueError(
                f"expected vision span length {EXPECTED_VISION_SPAN}, got {vision_span.size(1)}"
            )

        ext_dtype = next(self.extractor.parameters()).dtype
        x = vision_span.to(device=self.device, dtype=ext_dtype)
        logits, _ = self.extractor(x)  # [B, V]
        probs = torch.sigmoid(logits.float())

        threshold_tensor = torch.tensor(self.thresholds, dtype=probs.dtype, device=probs.device)
        onehot = (probs >= threshold_tensor.view(1, -1)).to(probs.dtype)

        probs_list = [float(v) for v in probs.reshape(-1).detach().cpu().tolist()]
        onehot_list = [int(v) for v in onehot.reshape(-1).detach().cpu().tolist()]
        active = [
            self.vocab[i] if i < len(self.vocab) else f"motif_{i}"
            for i, x in enumerate(onehot_list)
            if x
        ]
        return {
            "checkpoint": self.checkpoint,
            "threshold_source": self.threshold_source,
            "vocab": self.vocab,
            "logits": logits.detach().cpu(),
            "probabilities": probs_list,
            "motif_onehot": onehot_list,
            "active_motifs": active,
        }
