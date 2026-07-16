#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spec2mol.jdx import bundle_to_jsonable, load_spectra_dir
from spec2mol.model import Spec2MolModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Spec2Mol inference demo from JDX spectra.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config" / "spec2mol.yaml"))
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_dir", default=None, help="Override ckpt.dir in the config, e.g. spec2mol/ckpt/diffspectra.")
    parser.add_argument("--model_dir", default=None, help="Alias of --ckpt_dir for compatibility.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sample.json"
    sdf_path = output_dir / "sample.sdf"

    ckpt_dir = args.ckpt_dir or args.model_dir
    record = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "config": str(Path(args.config).resolve()),
        "ckpt_dir": str(Path(ckpt_dir).expanduser().resolve()) if ckpt_dir else None,
        "sdf_path": str(sdf_path),
        "ok": False,
    }
    try:
        bundle = load_spectra_dir(args.input_dir)
        record.update(bundle_to_jsonable(bundle))

        model = Spec2MolModel(args.config, device=args.device, ckpt_dir=ckpt_dir)
        result = model.generate(bundle)
        sdf_written = model.save_sdf(result, sdf_path)
        record.update(result.metadata)
        record.update(
            {
                "ok": sdf_written,
                "sdf_written": sdf_written,
                "smiles": result.smiles,
                "motif_onehot": result.motif_onehot,
                "motif_probabilities": result.motif_probabilities,
            }
        )
        if not sdf_written:
            record["failure_stage"] = "sdf_write"
            record["error"] = "RDKit molecule was not produced"
    except Exception as exc:
        record["failure_stage"] = record.get("failure_stage", "inference")
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 1

    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
