"""File-level assembly of a HF model with inline spec2mol tensors.

The 32B VLM shards are not loaded into memory. They are hardlinked into the
target directory with the shard suffix rewritten from ``of-00014`` to
``of-00015``. The small spec2mol modules are loaded from ``small_modules.pt``,
flattened under the ``spec2mol.`` tensor namespace, and saved as the final HF
shard, for example ``model-00015-of-00015.safetensors``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import torch

TENSOR_PREFIX = "spec2mol."
SAFETENSORS_GLOB = "model-*.safetensors"
SMALL_FILES_TO_COPY = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]
STALE_FILES = [
    "spec2mol_merge_config.json",
    "spec2mol_components_manifest.json",
    "spec2mol_modules.safetensors",
    "spec2mol_modules.safetensors.index.json",
]
STALE_TRAINING_FILES = ["trainer_state.json", "training_args.bin", "args.json"]
SHARD_RE = re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$")


def _install_pathlib_pickle_compat() -> None:
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


def _try_hardlink(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return True
    except OSError:
        return False


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def _clean_merged_dir(dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (SAFETENSORS_GLOB,):
        for path in dst_dir.glob(pattern):
            if path.exists() or path.is_symlink():
                path.unlink()
    for name in STALE_FILES + STALE_TRAINING_FILES + ["model.safetensors.index.json"]:
        path = dst_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()


def _copy_hf_side_files(src_dir: Path, dst_dir: Path) -> dict[str, dict[str, Any]]:
    copied: dict[str, dict[str, Any]] = {}
    for name in SMALL_FILES_TO_COPY:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[name] = {"source": str(src), "target": str(dst), "size_bytes": int(src.stat().st_size)}
    return copied


def _load_base_index(base_model_dir: Path) -> dict[str, Any]:
    index_path = base_model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    return json.loads(index_path.read_text(encoding="utf-8"))


def _shard_number(filename: str) -> int:
    match = SHARD_RE.match(filename)
    if not match:
        raise ValueError(f"unexpected HF shard filename: {filename}")
    return int(match.group(1))


def _rewrite_shard_name(filename: str, new_total: int) -> str:
    match = SHARD_RE.match(filename)
    if not match:
        raise ValueError(f"unexpected HF shard filename: {filename}")
    shard_idx = int(match.group(1))
    return f"model-{shard_idx:05d}-of-{new_total:05d}.safetensors"


def _link_vlm_shards(base_model_dir: Path, dst_dir: Path, base_index: dict[str, Any]) -> dict[str, str]:
    old_filenames = sorted(set(base_index.get("weight_map", {}).values()), key=_shard_number)
    if not old_filenames:
        raise ValueError("base model index has no safetensors weight_map entries")
    new_total = len(old_filenames) + 1
    rename_map = {name: _rewrite_shard_name(name, new_total) for name in old_filenames}
    for old_name, new_name in rename_map.items():
        src = base_model_dir / old_name
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = dst_dir / new_name
        if not _try_hardlink(src, dst):
            _symlink(src, dst)
    return rename_map


def _jsonable_module_entry(entry: dict[str, Any], hf_prefix: str, tensor_count: int) -> dict[str, Any]:
    keep = {}
    for key, value in entry.items():
        if key in {"state_dict", "source", "config", "config_path", "runtime_root"}:
            continue
        keep[key] = value
    keep["hf_prefix"] = hf_prefix
    keep["tensor_count"] = tensor_count
    return keep


def flatten_small_modules(small_modules_pt: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load ``small_modules.pt`` and flatten tensors under ``spec2mol.``."""
    small_pt = Path(small_modules_pt).expanduser().resolve()
    payload = _torch_load(small_pt)
    if not isinstance(payload, dict) or "modules" not in payload:
        raise ValueError(f"unexpected small_modules.pt format: {small_pt}")
    modules = payload["modules"]
    flat: dict[str, torch.Tensor] = {}
    config_mirror: dict[str, Any] = {
        "format": payload.get("format"),
        "tensor_prefix": TENSOR_PREFIX,
        "source_pt": str(small_pt),
        "modules": {},
    }
    for module_name, entry in modules.items():
        state = entry.get("state_dict")
        if not isinstance(state, dict):
            raise KeyError(f"{module_name} missing state_dict")
        local_prefix = str(entry.get("prefix") or f"{module_name}.")
        hf_prefix = f"{TENSOR_PREFIX}{local_prefix}"
        count = 0
        for key, tensor in state.items():
            out_key = f"{hf_prefix}{key}"
            if out_key in flat:
                raise ValueError(f"duplicate small-module tensor key: {out_key}")
            flat[out_key] = tensor.detach().cpu().contiguous()
            count += 1
        config_mirror["modules"][module_name] = _jsonable_module_entry(entry, hf_prefix, count)
    return flat, config_mirror


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _write_small_module_shard(
    flat_tensors: dict[str, torch.Tensor],
    dst_dir: Path,
    shard_filename: str,
    small_format: str | None,
) -> int:
    from safetensors.torch import save_file

    shard_path = dst_dir / shard_filename
    metadata = {
        "format": small_format or "spec2mol_small_modules",
        "tensor_prefix": TENSOR_PREFIX,
    }
    save_file(flat_tensors, str(shard_path), metadata=metadata)
    return sum(_tensor_nbytes(tensor) for tensor in flat_tensors.values())


def _write_index(
    base_index: dict[str, Any],
    rename_map: dict[str, str],
    flat_tensors: dict[str, torch.Tensor],
    small_shard: str,
    small_tensor_size: int,
    dst_dir: Path,
    small_config: dict[str, Any],
) -> dict[str, Any]:
    base_weight_map = base_index.get("weight_map", {})
    weight_map = {key: rename_map[value] for key, value in base_weight_map.items()}
    for key in flat_tensors:
        weight_map[key] = small_shard

    metadata = dict(base_index.get("metadata") or {})
    metadata["total_size"] = int(metadata.get("total_size") or 0) + int(small_tensor_size)
    metadata["spec2mol_format"] = "spec2mol_v2_hf_inline_safetensors"
    metadata["spec2mol_tensor_prefix"] = TENSOR_PREFIX
    metadata["spec2mol_shard"] = small_shard
    metadata["spec2mol_small_modules_format"] = small_config.get("format")
    metadata["spec2mol_small_module_tensors"] = len(flat_tensors)

    index = {"metadata": metadata, "weight_map": weight_map}
    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return index


def _write_config(dst_dir: Path, small_config: dict[str, Any], small_shard: str) -> None:
    config_path = dst_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    hf_config = json.loads(config_path.read_text(encoding="utf-8"))
    hf_config["spec2mol_v2"] = {
        "format": "spec2mol_v2_hf_inline_safetensors",
        "tensor_prefix": TENSOR_PREFIX,
        "small_modules": {
            **small_config,
            "source": "model.safetensors.index.json",
            "shard": small_shard,
        },
        "runtime": {
            "module_source": "hf_safetensors",
            "index_file": "model.safetensors.index.json",
            "tensor_prefix": TENSOR_PREFIX,
            "predictors": {
                "ir": "hf_safetensors:spec2mol.s1_omni_ir_linear.",
                "raman": "hf_safetensors:spec2mol.s1_omni_raman_linear.",
                "uv": "hf_safetensors:spec2mol.s1_omni_uv_linear.",
            },
            "route_image_order": ["ir", "raman", "uv"],
            "modality_prompts": {
                "ir": "请你分析这张IR谱图",
                "raman": "请你分析这张Raman谱图",
                "uv": "请你分析这张UV谱图",
            },
            "missing_modality_policy": {
                "predictor": "skip_missing",
                "spcformer_context": "zero_fill_missing",
            },
            "motif_fusion": "onehot_intersection",
            "spcformer_context_order": ["uv", "ir", "raman"],
            "motif_source": "s1_omni_linear_present_modalities",
            "specformer_usage": ["dmt_context", "feat256", "atom_count"],
            "specformer_motif_head_used": False,
        },
    }
    config_path.write_text(json.dumps(hf_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_merged_dir(
    base_model_dir: str | Path,
    small_modules_pt: str | Path,
    merged_dir: str | Path,
) -> dict[str, Any]:
    base_dir = Path(base_model_dir).expanduser().resolve()
    small_pt = Path(small_modules_pt).expanduser().resolve()
    dst_dir = Path(merged_dir).expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(base_dir)
    if not small_pt.is_file():
        raise FileNotFoundError(small_pt)

    base_index = _load_base_index(base_dir)
    _clean_merged_dir(dst_dir)
    copied = _copy_hf_side_files(base_dir, dst_dir)
    rename_map = _link_vlm_shards(base_dir, dst_dir, base_index)
    new_total = len(rename_map) + 1
    small_shard = f"model-{new_total:05d}-of-{new_total:05d}.safetensors"

    flat_tensors, small_config = flatten_small_modules(small_pt)
    small_tensor_size = _write_small_module_shard(
        flat_tensors,
        dst_dir,
        small_shard,
        small_config.get("format"),
    )
    index = _write_index(
        base_index,
        rename_map,
        flat_tensors,
        small_shard,
        small_tensor_size,
        dst_dir,
        small_config,
    )
    _write_config(dst_dir, small_config, small_shard)

    return {
        "merged_dir": str(dst_dir),
        "base_model_dir": str(base_dir),
        "small_modules_pt": str(small_pt),
        "copied_files": copied,
        "vlm_shards": rename_map,
        "small_module_shard": small_shard,
        "small_module_tensors": len(flat_tensors),
        "total_shards": new_total,
        "index": str(dst_dir / "model.safetensors.index.json"),
        "config": str(dst_dir / "config.json"),
        "index_total_tensors": len(index["weight_map"]),
    }
