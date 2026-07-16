"""Three-modality S1-Omni linear motif predictors.

Each predictor consumes one modality's Qwen3-VL vision-span hidden state with
shape ``[B, 546, 5120]`` and returns 20 motif logits/probabilities/onehot bits.
Multiple modalities are fused by onehot intersection.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import torch

from spec2mol_v2.prefix_attn_extractor import PrefixAttnExtractor
from spec2mol_v2.vision_span import EXPECTED_VISION_SPAN

MODALITIES = ("ir", "raman", "uv")
DEFAULT_PROMPTS = {
    "ir": "请你分析这张IR谱图",
    "raman": "请你分析这张Raman谱图",
    "uv": "请你分析这张UV谱图",
}
DEFAULT_VOCAB = [
    "alkane",
    "alkene",
    "alkyne",
    "amine",
    "imine",
    "nitrile",
    "alcohol",
    "ether",
    "haloalkane",
    "aldehyde",
    "ketone",
    "ester",
    "amide",
    "arene",
    "imidazole",
    "pyrazole",
    "oxazole",
    "isoxazole",
    "cyclopropane",
    "epoxide",
]


def _install_pathlib_pickle_compat() -> None:
    # Some staged checkpoints were saved from a Python build that pickled
    # pathlib classes under pathlib._local. Python 3.10 does not expose that
    # module, so provide a minimal alias before torch.load unpickles metadata.
    if "pathlib._local" in sys.modules:
        return
    module = types.ModuleType("pathlib._local")
    module.Path = Path
    module.PosixPath = type(Path("."))
    module.WindowsPath = type(Path("C:/"))
    sys.modules["pathlib._local"] = module


def _torch_load(path: Path) -> Any:
    _install_pathlib_pickle_compat()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_config(blob: dict[str, Any], config_json: dict[str, Any]) -> dict[str, Any]:
    cfg = blob.get("model_config") or config_json.get("model_config") or {}
    if not cfg:
        args = blob.get("args") or config_json
        cfg = {
            "in_dim": int(args.get("input_dim", args.get("in_dim", 5120))),
            "proj_dim": int(args.get("proj_dim", 768)),
            "n_layers": int(args.get("n_layers", 2)),
            "n_heads": int(args.get("n_heads", 8)),
            "ffn_mult": int(args.get("ffn_mult", 2)),
            "dropout": float(args.get("dropout", 0.1)),
            "num_queries": int(args.get("num_queries", 4)),
            "num_classes": int(args.get("num_labels", args.get("num_classes", 20))),
            "feat_dim": int(args.get("feat_dim", 0)),
            "head_dropout": float(args.get("head_dropout", 0.1)),
        }
    cfg = dict(cfg)
    cfg.setdefault("in_dim", 5120)
    cfg.setdefault("proj_dim", 768)
    cfg.setdefault("n_layers", 2)
    cfg.setdefault("n_heads", 8)
    cfg.setdefault("ffn_mult", 2)
    cfg.setdefault("dropout", 0.1)
    cfg.setdefault("num_queries", 4)
    cfg.setdefault("num_classes", 20)
    cfg.setdefault("feat_dim", 0)
    cfg.setdefault("head_dropout", 0.1)
    return cfg


def _state_dict(blob: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = blob.get("model_state_dict") or blob.get("state_dict")
    if not isinstance(state, dict):
        raise KeyError("checkpoint must contain model_state_dict or state_dict")
    return {k: v.detach().cpu().contiguous() for k, v in state.items()}


def _thresholds(blob: dict[str, Any], config_json: dict[str, Any], num_classes: int) -> tuple[list[float], str]:
    raw = blob.get("thresholds_tuned") or blob.get("thresholds_default")
    source = "thresholds_tuned" if blob.get("thresholds_tuned") is not None else "thresholds_default"
    if raw is None:
        raw = config_json.get("thresholds_tuned") or config_json.get("thresholds_default")
        source = "thresholds_tuned" if config_json.get("thresholds_tuned") is not None else "thresholds_default"
    if raw is None:
        return [0.5] * num_classes, "0.5"
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().tolist()
    values = [float(x) for x in raw]
    if len(values) < num_classes:
        values.extend([0.5] * (num_classes - len(values)))
    return values[:num_classes], source


class S1OmniLinearPredictor:
    """One modality-specific S1-Omni PrefixAttnExtractor."""

    def __init__(
        self,
        checkpoint: str | Path,
        modality: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if modality not in MODALITIES:
            raise ValueError(f"unsupported modality {modality!r}; expected one of {MODALITIES}")
        self.modality = modality
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        ckpt = Path(self.checkpoint)
        blob = _torch_load(ckpt)
        if not isinstance(blob, dict):
            raise ValueError(f"unexpected checkpoint format: {ckpt}")
        config_json = _load_json(ckpt.with_name("config.json"))
        build = _model_config(blob, config_json)
        self.num_classes = int(build["num_classes"])
        self.vocab = list(blob.get("vocab") or config_json.get("vocab") or DEFAULT_VOCAB[: self.num_classes])
        if len(self.vocab) < self.num_classes:
            self.vocab.extend(f"motif_{i}" for i in range(len(self.vocab), self.num_classes))
        self.thresholds, self.threshold_source = _thresholds(blob, config_json, self.num_classes)
        self.build_args = build

        model = PrefixAttnExtractor(
            in_dim=int(build["in_dim"]),
            proj_dim=int(build["proj_dim"]),
            n_layers=int(build["n_layers"]),
            n_heads=int(build["n_heads"]),
            ffn_mult=int(build["ffn_mult"]),
            dropout=float(build["dropout"]),
            num_queries=int(build["num_queries"]),
            num_classes=self.num_classes,
            feat_dim=int(build.get("feat_dim", 0)),
            head_dropout=float(build.get("head_dropout", 0.1)),
            frozen_head_state=None,
        )
        model.load_state_dict(_state_dict(blob), strict=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        target_dtype = dtype if dtype is not None else next(model.parameters()).dtype
        self.model = model.to(device=self.device, dtype=target_dtype)
        self.model.eval()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        modality: str,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "S1OmniLinearPredictor":
        if modality not in MODALITIES:
            raise ValueError(f"unsupported modality {modality!r}; expected one of {MODALITIES}")
        config = config or {}
        build = dict(config.get("build_args") or {})
        build.setdefault("in_dim", 5120)
        build.setdefault("proj_dim", 768)
        build.setdefault("n_layers", 2)
        build.setdefault("n_heads", 8)
        build.setdefault("ffn_mult", 2)
        build.setdefault("dropout", 0.1)
        build.setdefault("num_queries", 4)
        build.setdefault("num_classes", 20)
        build.setdefault("feat_dim", 0)
        build.setdefault("head_dropout", 0.1)

        self = cls.__new__(cls)
        self.modality = modality
        self.checkpoint = f"hf_safetensors:{config.get('hf_prefix', f'spec2mol.s1_omni_{modality}_linear.')}"
        self.num_classes = int(build["num_classes"])
        self.vocab = list(config.get("vocab") or DEFAULT_VOCAB[: self.num_classes])
        if len(self.vocab) < self.num_classes:
            self.vocab.extend(f"motif_{i}" for i in range(len(self.vocab), self.num_classes))
        raw_thresholds = config.get("thresholds")
        self.thresholds = [float(x) for x in (raw_thresholds or [0.5] * self.num_classes)][: self.num_classes]
        if len(self.thresholds) < self.num_classes:
            self.thresholds.extend([0.5] * (self.num_classes - len(self.thresholds)))
        self.threshold_source = str(config.get("threshold_source") or "0.5")
        self.build_args = build

        model = PrefixAttnExtractor(
            in_dim=int(build["in_dim"]),
            proj_dim=int(build["proj_dim"]),
            n_layers=int(build["n_layers"]),
            n_heads=int(build["n_heads"]),
            ffn_mult=int(build["ffn_mult"]),
            dropout=float(build["dropout"]),
            num_queries=int(build["num_queries"]),
            num_classes=self.num_classes,
            feat_dim=int(build.get("feat_dim", 0)),
            head_dropout=float(build.get("head_dropout", 0.1)),
            frozen_head_state=None,
        )
        clean_state = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        target_dtype = dtype if dtype is not None else next(model.parameters()).dtype
        self.model = model.to(device=self.device, dtype=target_dtype)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, vision_span: torch.Tensor) -> dict[str, Any]:
        if vision_span.dim() != 3:
            raise ValueError(f"expected vision_span [B, 546, 5120], got {tuple(vision_span.shape)}")
        if int(vision_span.size(1)) != EXPECTED_VISION_SPAN:
            raise ValueError(f"expected span length {EXPECTED_VISION_SPAN}, got {vision_span.size(1)}")
        dtype = next(self.model.parameters()).dtype
        logits, _ = self.model(vision_span.to(device=self.device, dtype=dtype))
        probs = torch.sigmoid(logits.float())
        threshold = torch.tensor(self.thresholds, device=probs.device, dtype=probs.dtype).view(1, -1)
        onehot = (probs >= threshold).to(torch.float32)
        probs_list = [float(x) for x in probs.reshape(-1).detach().cpu().tolist()]
        onehot_list = [int(x) for x in onehot.reshape(-1).detach().cpu().tolist()]
        return {
            "modality": self.modality,
            "checkpoint": self.checkpoint,
            "threshold_source": self.threshold_source,
            "thresholds": self.thresholds,
            "vocab": self.vocab,
            "logits": logits.detach().cpu(),
            "probabilities": probs_list,
            "motif_onehot": onehot_list,
            "active_motifs": [self.vocab[i] if i < len(self.vocab) else f"motif_{i}" for i, x in enumerate(onehot_list) if x],
        }


class S1OmniLinearPredictorBank:
    """Load and run the IR/Raman/UV predictor bank."""

    def __init__(self, checkpoints: dict[str, str | Path], device: str | torch.device | None = None) -> None:
        self.predictors = {
            modality: S1OmniLinearPredictor(path, modality, device=device)
            for modality, path in checkpoints.items()
            if modality in MODALITIES and path
        }
        if not self.predictors:
            raise ValueError("no S1-Omni linear predictors configured")

    @classmethod
    def from_state_dicts(
        cls,
        modules: dict[str, dict[str, Any]],
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "S1OmniLinearPredictorBank":
        name_map = {
            "ir": "s1_omni_ir_linear",
            "raman": "s1_omni_raman_linear",
            "uv": "s1_omni_uv_linear",
        }
        self = cls.__new__(cls)
        predictors = {}
        for modality, module_name in name_map.items():
            entry = modules.get(module_name)
            if not entry:
                continue
            cfg = dict(entry.get("config") or {})
            if entry.get("hf_prefix"):
                cfg.setdefault("hf_prefix", entry["hf_prefix"])
            predictors[modality] = S1OmniLinearPredictor.from_state_dict(
                entry["state_dict"],
                modality=modality,
                config=cfg,
                device=device,
                dtype=dtype,
            )
        self.predictors = predictors
        if not self.predictors:
            raise ValueError("no S1-Omni linear predictor states configured")
        return self

    @property
    def vocab(self) -> list[str]:
        first = next(iter(self.predictors.values()))
        return first.vocab

    @torch.no_grad()
    def predict(self, modality_to_span: dict[str, torch.Tensor]) -> dict[str, Any]:
        per_modality: dict[str, Any] = {}
        for modality in MODALITIES:
            if modality not in modality_to_span:
                continue
            if modality not in self.predictors:
                raise KeyError(f"missing predictor for present modality {modality}")
            per_modality[modality] = self.predictors[modality].predict(modality_to_span[modality])
        if not per_modality:
            raise ValueError("no present modality vision spans were provided")
        onehots = [torch.tensor(v["motif_onehot"], dtype=torch.bool) for v in per_modality.values()]
        probs = [torch.tensor(v["probabilities"], dtype=torch.float32) for v in per_modality.values()]
        fused_onehot = torch.stack(onehots, dim=0).all(dim=0).to(torch.int64)
        fused_probs = torch.stack(probs, dim=0).min(dim=0).values
        onehot_list = [int(x) for x in fused_onehot.tolist()]
        probs_list = [float(x) for x in fused_probs.tolist()]
        vocab = self.vocab
        return {
            "motif_fusion": "onehot_intersection",
            "modalities_used": list(per_modality.keys()),
            "per_modality": per_modality,
            "vocab": vocab,
            "probabilities": probs_list,
            "motif_onehot": onehot_list,
            "active_motifs": [vocab[i] if i < len(vocab) else f"motif_{i}" for i, x in enumerate(onehot_list) if x],
        }
