#!/usr/bin/env python3
"""End-to-end merged-model inference for one JDX sample.

Pipeline:
  JDX -> render present spectra PNGs -> VLM route over present spectra
  -> modality-specific VLM hidden -> S1-Omni linear predictors
  -> onehot intersection motif -> spcformer zero-filled context + DMT -> SDF
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spec2mol_v2.model import Spec2MolModel
from spec2mol_v2.s1_omni_linear_predictor import DEFAULT_PROMPTS, MODALITIES, S1OmniLinearPredictorBank
from spec2mol_v2.spectra_plot import render_spectra_images
from spec2mol_v2.vlm_runtime import VLMRuntime

DEFAULT_MERGED_MODEL_DIR = ""
DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "config" / "spec2mol_v2.yaml")
DEFAULT_INPUT_DIR = str(PROJECT_ROOT / "spec2mol" / "dataset" / "test_data" / "000010")
CONTEXT_INDEX = {"uv": 0, "ir": 1, "raman": 2}


def load_spec2mol_hf_config(merged_model_dir: Path) -> dict[str, Any]:
    config_path = merged_model_dir / "config.json"
    if config_path.exists():
        hf_config = json.loads(config_path.read_text(encoding="utf-8"))
        inline = hf_config.get("spec2mol_v2")
        if isinstance(inline, dict) and isinstance(inline.get("runtime"), dict):
            return inline
    legacy_path = merged_model_dir / "spec2mol_merge_config.json"
    if legacy_path.exists():
        return {
            "format": "spec2mol_v2_legacy_sidecar",
            "runtime": json.loads(legacy_path.read_text(encoding="utf-8")),
            "components": None,
        }
    raise FileNotFoundError(
        f"spec2mol_v2 config not found in {config_path} and legacy sidecar missing: {legacy_path}"
    )


def load_component_manifest(component_root: Path, inline_components: dict[str, Any] | None = None) -> dict[str, Any]:
    if inline_components and inline_components.get("components"):
        return inline_components
    path = component_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"component manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_zero_filled_context(bundle) -> dict[str, Any]:
    details = {}
    ok = True
    for modality in bundle.zero_filled_spectra:
        idx = CONTEXT_INDEX[modality]
        tensor = bundle.context[idx]
        is_zero = bool(torch.count_nonzero(tensor).item() == 0)
        details[modality] = {"shape": list(tensor.shape), "all_zero": is_zero}
        ok = ok and is_zero
    return {"ok": ok, "details": details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="spec2mol_v2 merged-model inference for one sample.")
    parser.add_argument("--merged_model_dir", default=DEFAULT_MERGED_MODEL_DIR)
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--prompt", default=None, help="Override route prompt")
    parser.add_argument("--ir_prompt", default=None)
    parser.add_argument("--raman_prompt", default=None)
    parser.add_argument("--uv_prompt", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_image_side", type=int, default=None)
    parser.add_argument("--force_spectra", action="store_true")
    parser.add_argument("--skip_vlm_generate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model_dir = Path(args.merged_model_dir).expanduser().resolve()
    sample_json = output_dir / "sample.json"
    sdf_path = output_dir / "sample.sdf"

    record: dict[str, Any] = {
        "merged_model_dir": str(merged_model_dir),
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(output_dir),
        "config": str(Path(args.config).resolve()),
        "sdf_path": str(sdf_path),
        "ok": False,
        "failure_stage": None,
        "error": None,
        "traceback": None,
    }

    try:
        spec2mol_cfg = load_spec2mol_hf_config(merged_model_dir)
        merge_cfg = spec2mol_cfg["runtime"]
        record["merged_model_config"] = merge_cfg
        record["spec2mol_config_format"] = spec2mol_cfg.get("format")
        record["spec2mol_config_source"] = "config.json:spec2mol_v2" if spec2mol_cfg.get("format") != "spec2mol_v2_legacy_sidecar" else "legacy_sidecar"
        component_root = Path(merge_cfg["component_root"]).resolve()
        manifest = load_component_manifest(component_root, spec2mol_cfg.get("components"))
        record["component_root"] = str(component_root)
        record["component_manifest"] = str(component_root / "manifest.json")
        predictors = {k: Path(v) for k, v in merge_cfg.get("predictors", {}).items()}
        if not predictors:
            components = manifest["components"]
            predictors = {
                "ir": Path(components["s1_omni_ir_linear"]["target"]),
                "raman": Path(components["s1_omni_raman_linear"]["target"]),
                "uv": Path(components["s1_omni_uv_linear"]["target"]),
            }
        dmt_ckpt_dir = Path(merge_cfg["dmt_ckpt_dir"])
        record["predictor_checkpoints"] = {k: str(v) for k, v in predictors.items()}
        record["dmt_checkpoint_or_ckpt_dir"] = str(dmt_ckpt_dir)
        record["vlm_loaded_from"] = str(merged_model_dir)

        from spec2mol_v2.jdx import bundle_to_jsonable, load_spectra_dir

        bundle = load_spectra_dir(args.input_dir)
        record["present_spectra"] = bundle.present_spectra
        record["zero_filled_spectra"] = bundle.zero_filled_spectra
        record["spectra_files"] = bundle_to_jsonable(bundle)["files"]
        zero_check = verify_zero_filled_context(bundle)
        record["spcformer_zero_fill_verified"] = zero_check["ok"]
        record["spcformer_zero_fill_details"] = zero_check["details"]
        if not zero_check["ok"]:
            record["failure_stage"] = "spcformer_zero_fill"
            record["error"] = "missing spectra context is not zero-filled"
            write_json(sample_json, record)
            return 1

        spectra_dir = output_dir / "spectra_images"
        image_meta = render_spectra_images(args.input_dir, spectra_dir, max_image_side=args.max_image_side)
        all_images = {k: Path(v) for k, v in image_meta.get("image_files", {}).items()}
        present_images = {k: all_images[k] for k in bundle.present_spectra if k in all_images}
        record["spectra_images"] = {k: str(v) for k, v in present_images.items()}
        record["image_metadata"] = {k: v for k, v in image_meta.items() if k != "image_files"}

        prompts = dict(merge_cfg.get("modality_prompts") or DEFAULT_PROMPTS)
        if args.ir_prompt:
            prompts["ir"] = args.ir_prompt
        if args.raman_prompt:
            prompts["raman"] = args.raman_prompt
        if args.uv_prompt:
            prompts["uv"] = args.uv_prompt
        record["per_modality_prompt"] = {k: prompts[k] for k in bundle.present_spectra if k in prompts}

        runtime = None
        route_info: dict[str, Any] = {"skipped": False}
        if args.skip_vlm_generate:
            route_info = {"skipped": True, "has_spectra_token": True, "image_order": list(present_images)}
            record["generated_text"] = None
            record["has_spectra_token"] = True
        else:
            runtime = VLMRuntime(merged_model_dir=merged_model_dir, device=args.device or "cuda:0", dtype=args.dtype)
            route = runtime.generate_route(
                modality_images=present_images,
                prompt=args.prompt or merge_cfg.get("route_prompt") or None,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            (output_dir / "vlm_generate.txt").write_text(route["generated_text"], encoding="utf-8")
            write_json(output_dir / "route.json", route)
            route_info = route
            record["generated_text"] = route["generated_text"]
            record["has_spectra_token"] = route["has_spectra_token"]

        write_json(sample_json, record)
        hit = bool(route_info.get("has_spectra_token")) or args.force_spectra
        if not hit:
            record["failure_stage"] = "route_no_spectra"
            record["error"] = "VLM route did not emit <spectra_st>; skipping motif + DMT."
            write_json(sample_json, record)
            print(json.dumps({"ok": False, "failure_stage": record["failure_stage"]}, indent=2))
            return 0

        if runtime is None:
            runtime = VLMRuntime(merged_model_dir=merged_model_dir, device=args.device or "cuda:0", dtype=args.dtype)

        spans: dict[str, torch.Tensor] = {}
        hidden_meta: dict[str, Any] = {}
        for modality in MODALITIES:
            if modality not in present_images:
                continue
            hidden = runtime.extract_vision_hidden(present_images[modality], modality=modality, prompt=prompts[modality])
            hidden_meta[modality] = {k: v for k, v in hidden.items() if k not in ("vision_span", "ir_vision_span")}
            if not hidden["ok"]:
                record["failure_stage"] = f"vision_span_shape_{modality}"
                record["error"] = hidden["error"] or "vision span shape mismatch"
                record["vision_hidden"] = hidden_meta
                write_json(sample_json, record)
                return 0
            spans[modality] = hidden["vision_span"]
        record["vision_hidden"] = hidden_meta

        bank = S1OmniLinearPredictorBank(predictors, device=args.device or "cuda:0")
        motif_out = bank.predict(spans)
        record["predictor_modalities_used"] = motif_out["modalities_used"]
        record["motif_fusion"] = motif_out["motif_fusion"]
        record["per_modality_motif"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "logits"}
            for k, v in motif_out["per_modality"].items()
        }
        record["motif_vocab"] = motif_out["vocab"]
        record["motif_probabilities"] = motif_out["probabilities"]
        record["motif_onehot"] = motif_out["motif_onehot"]
        record["motif_active"] = motif_out["active_motifs"]

        route_metadata = dict(route_info)
        route_metadata["motif_fusion"] = motif_out["motif_fusion"]
        route_metadata["predictor_modalities_used"] = motif_out["modalities_used"]
        route_metadata["per_modality_prompt"] = record["per_modality_prompt"]

        model = Spec2MolModel(args.config, device=args.device, ckpt_dir=args.ckpt_dir or str(dmt_ckpt_dir))
        result = model.generate_with_external_motif(
            bundle=bundle,
            motif_onehot=motif_out["motif_onehot"],
            motif_probabilities=motif_out["probabilities"],
            route_metadata=route_metadata,
        )
        record["atom_count"] = result.atom_count
        record["motif_onehot"] = result.motif_onehot
        record["motif_probabilities"] = result.motif_probabilities
        record["smiles"] = result.smiles
        record["motif_metadata"] = result.metadata
        sdf_written = model.save_sdf(result, sdf_path)
        record["sdf_written"] = sdf_written
        record["ok"] = sdf_written
        if not sdf_written:
            record["failure_stage"] = "sdf_write"
            record["error"] = "RDKit molecule was not produced"
        write_json(sample_json, record)
        print(json.dumps({"ok": record["ok"], "sdf_path": str(sdf_path), "smiles": result.smiles}, indent=2))
        return 0 if record["ok"] else 1

    except Exception as exc:  # noqa: BLE001
        record["failure_stage"] = record.get("failure_stage") or "inference"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
        write_json(sample_json, record)
        print(json.dumps({"ok": False, "failure_stage": record["failure_stage"], "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
