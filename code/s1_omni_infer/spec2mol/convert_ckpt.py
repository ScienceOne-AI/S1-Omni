#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spec2mol.utils import load_yaml, resolve_ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Spec2Mol PyTorch checkpoints to safetensors.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config" / "spec2mol.yaml"))
    return parser.parse_args()


def require_safetensors():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("safetensors is required: pip install safetensors") from exc
    return save_file


def _clone_cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}


def _ema_model_state(model_state: dict[str, torch.Tensor], ema_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    shadow_params = ema_state.get("shadow_params")
    if shadow_params is None:
        raise KeyError("EMA state missing shadow_params")
    out: dict[str, torch.Tensor] = {}
    shadow_iter = iter(shadow_params)
    for key, value in model_state.items():
        if torch.is_floating_point(value):
            try:
                out[key] = next(shadow_iter).detach().cpu().contiguous()
            except StopIteration as exc:
                raise ValueError("EMA shadow_params shorter than model floating parameters") from exc
        else:
            out[key] = value.detach().cpu().contiguous()
    return out


def convert_dmt(cfg: dict[str, Any], save_file) -> dict[str, Any]:
    source_path = Path(cfg["source"]["dmt_checkpoint"])
    target_path = resolve_ckpt_path(cfg, "dmt")

    from spec2mol.utils import add_llm_dmt_to_path, load_py_config

    add_llm_dmt_to_path(cfg["llm_dmt_root"])
    from models import create_model  # type: ignore
    from models.ema import ExponentialMovingAverage  # type: ignore

    dmt_config = load_py_config(cfg["source"]["config"])
    dmt_config.training.distributed = False
    dmt_config.training.num_gpus = 1
    dmt_config.training.world_size = 1
    dmt_config.training.local_rank = 0
    dmt_config.device = torch.device("cpu")
    dmt_config.data.spectra_version = cfg["model"].get("spectra_version", "allspectra")

    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    model = create_model(dmt_config)
    model.load_state_dict(payload["model"], strict=True)
    source_state = "model"
    if cfg["model"].get("use_ema", True):
        ema = ExponentialMovingAverage(model.parameters(), decay=dmt_config.model.ema_decay)
        ema.load_state_dict(payload["ema"])
        ema.copy_to(model.parameters())
        source_state = "ema"

    state = _clone_cpu_state_dict(model.state_dict())
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(target_path))
    return {
        "source": str(source_path),
        "target": str(target_path),
        "source_state": source_state,
        "num_tensors": len(state),
        "step": int(payload.get("step", -1)),
    }


def convert_head(cfg: dict[str, Any], source_key: str, target_key: str, save_file) -> dict[str, Any]:
    source_path = Path(cfg["source"][source_key])
    target_path = resolve_ckpt_path(cfg, target_key)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    state = _clone_cpu_state_dict(payload["state_dict"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(target_path))
    meta = {
        "source": str(source_path),
        "target": str(target_path),
        "num_tensors": len(state),
    }
    for key in ("args", "in_dim", "num_classes", "task", "mean", "std", "hidden", "dropout", "vocab"):
        if key in payload:
            meta[key] = payload[key]
    for key in ("thresholds_tuned", "thresholds_default"):
        if key in payload:
            value = payload[key]
            meta[key] = value.detach().cpu().tolist() if torch.is_tensor(value) else value
    return meta


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    save_file = require_safetensors()

    metadata = {
        "config": str(Path(args.config).resolve()),
        "dmt": convert_dmt(cfg, save_file),
        "atomcount": convert_head(cfg, "atomcount_checkpoint", "atomcount", save_file),
        "motif": convert_head(cfg, "motif_checkpoint", "motif", save_file),
        "model": cfg["model"],
        "jdx": cfg["jdx"],
    }
    metadata_path = resolve_ckpt_path(cfg, "metadata")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
