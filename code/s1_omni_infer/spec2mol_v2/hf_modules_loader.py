"""Load inline spec2mol tensors from a merged HuggingFace checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

TENSOR_PREFIX = "spec2mol."
MODULE_PREFIXES = {
    "s1_omni_ir_linear": "spec2mol.s1_omni_ir_linear.",
    "s1_omni_raman_linear": "spec2mol.s1_omni_raman_linear.",
    "s1_omni_uv_linear": "spec2mol.s1_omni_uv_linear.",
    "atomcount": "spec2mol.atomcount.",
    "motif": "spec2mol.motif.",
    "dmt": "spec2mol.dmt.",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_hf_inline_modules(merged_model_dir: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Return module state_dicts/configs embedded in merged HF safetensors."""
    merged_dir = Path(merged_model_dir).expanduser().resolve()
    config = _load_json(merged_dir / "config.json")
    spec_cfg = config.get("spec2mol_v2")
    if not isinstance(spec_cfg, dict) or spec_cfg.get("format") != "spec2mol_v2_hf_inline_safetensors":
        raise ValueError(f"{merged_dir} is not a spec2mol HF-inline checkpoint")

    index = _load_json(merged_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map") or {}
    shard_names = sorted({filename for key, filename in weight_map.items() if key.startswith(TENSOR_PREFIX)})
    if not shard_names:
        raise ValueError(f"no {TENSOR_PREFIX} tensors found in {merged_dir / 'model.safetensors.index.json'}")

    from safetensors.torch import load_file

    loaded: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = merged_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        shard_state = load_file(str(shard_path), device=device)
        for key, value in shard_state.items():
            if key.startswith(TENSOR_PREFIX):
                loaded[key] = value

    modules_config = ((spec_cfg.get("small_modules") or {}).get("modules") or {})
    modules: dict[str, dict[str, Any]] = {}
    for module_name, prefix in MODULE_PREFIXES.items():
        state = {
            key[len(prefix) :]: tensor
            for key, tensor in loaded.items()
            if key.startswith(prefix)
        }
        if not state:
            raise KeyError(f"missing tensors for {module_name} ({prefix})")
        modules[module_name] = {
            "state_dict": state,
            "config": modules_config.get(module_name, {}),
            "hf_prefix": prefix,
        }

    return {
        "merged_model_dir": str(merged_dir),
        "format": spec_cfg.get("format"),
        "tensor_prefix": TENSOR_PREFIX,
        "modules": modules,
        "config": spec_cfg,
    }
