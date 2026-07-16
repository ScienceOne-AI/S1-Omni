from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spec2mol.convert_ckpt import (  # noqa: F401
    convert_dmt,
    convert_head,
    require_safetensors,
)
from spec2mol.utils import load_yaml, resolve_ckpt_path  # noqa: F401

DEFAULT_V2_CONFIG = str(Path(__file__).resolve().parent / "config" / "spec2mol_v2.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Spec2Mol PyTorch checkpoints to safetensors (v2).")
    parser.add_argument("--config", default=DEFAULT_V2_CONFIG)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    save_file = require_safetensors()

    dmt_meta = convert_dmt(cfg, save_file)
    atom_meta = convert_head(cfg, "atomcount_checkpoint", "atomcount", save_file)
    motif_meta = convert_head(cfg, "motif_checkpoint", "motif", save_file)
    for entry in (dmt_meta, atom_meta, motif_meta):
        entry.pop("source", None)
    metadata = {
        "config": str(Path(args.config).resolve()),
        "dmt": dmt_meta,
        "atomcount": atom_meta,
        "motif": motif_meta,
        "motif_head_unused": True,
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
