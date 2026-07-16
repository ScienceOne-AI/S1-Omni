from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import yaml


SPEC2MOL_ROOT = Path(__file__).resolve().parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_ckpt_path(cfg: dict[str, Any], key: str) -> Path:
    ckpt_cfg = cfg["ckpt"]
    return Path(ckpt_cfg["dir"]) / ckpt_cfg[key]


def add_llm_dmt_to_path(llm_dmt_root: str | Path) -> Path:
    root = Path(llm_dmt_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_py_config(config_path: str | Path):
    config_path = Path(config_path)
    spec = importlib.util.spec_from_file_location("spec2mol_llm_dmt_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()
