#!/usr/bin/env python3
"""Inference from a HF checkpoint with inline spec2mol tensors."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spec2mol_v2.hf_modules_loader import load_hf_inline_modules
from spec2mol_v2.model import Spec2MolModel
from spec2mol_v2.s1_omni_linear_predictor import DEFAULT_PROMPTS, MODALITIES, S1OmniLinearPredictorBank
from spec2mol_v2.spectra_plot import render_spectra_images
from spec2mol_v2.vlm_runtime import ROUTE_DEFAULT_PROMPT, VLMRuntime

DEFAULT_MERGED_MODEL_DIR = ""
DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "config" / "spec2mol_v2.yaml")
CONTEXT_INDEX = {"uv": 0, "ir": 1, "raman": 2}


def _resolve_dtype(dtype: str | None) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = dtype.lower()
    table = {
        None: torch.bfloat16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return table.get(dtype, torch.bfloat16)


def _stage_jdx_files(jdx_files: list[str], temp_root: Path) -> Path:
    input_dir = temp_root / "jdx"
    input_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for idx, raw in enumerate(jdx_files, start=1):
        src = Path(raw).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        name = src.name
        if name in seen:
            name = f"{idx:02d}_{name}"
        seen.add(name)
        dst = input_dir / name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return input_dir


def verify_zero_filled_context(bundle) -> None:
    for modality in bundle.zero_filled_spectra:
        idx = CONTEXT_INDEX[modality]
        tensor = bundle.context[idx]
        if torch.count_nonzero(tensor).item() != 0:
            raise RuntimeError(f"missing {modality} spectra context is not zero-filled")


def _make_final_text(generated_text: str, sdf_path: Path | None, force_spectra: bool) -> str:
    if sdf_path is None:
        return generated_text
    replacement = str(sdf_path.resolve())
    if "<spectra_st>" in generated_text:
        return generated_text.replace("<spectra_st>", replacement)
    if force_spectra:
        return replacement
    return generated_text


def _terminal_json(question: str, generated_text: str, final_text: str) -> str:
    return json.dumps(
        {
            "question": question,
            "generated_text": generated_text,
            "final_text": final_text,
        },
        ensure_ascii=False,
        indent=2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged_model_dir", default=DEFAULT_MERGED_MODEL_DIR)
    parser.add_argument("--jdx_files", nargs="+", required=True, help="one or more IR/Raman/UV JDX files")
    parser.add_argument("--output_dir", required=True, help="directory for sample.sdf only when generated")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", default=ROUTE_DEFAULT_PROMPT)
    parser.add_argument("--ir_prompt", default=None)
    parser.add_argument("--raman_prompt", default=None)
    parser.add_argument("--uv_prompt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
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
    for child in output_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    sdf_path = output_dir / "sample.sdf"

    question = args.prompt or ROUTE_DEFAULT_PROMPT
    generated_text = ""
    final_sdf_path: Path | None = None

    try:
        merged_model_dir = Path(args.merged_model_dir).expanduser().resolve()
        inline = load_hf_inline_modules(merged_model_dir, device="cpu")
        modules = inline["modules"]

        from spec2mol_v2.jdx import load_spectra_dir

        with tempfile.TemporaryDirectory(prefix="spec2mol_hf_infer_") as temp_name:
            temp_root = Path(temp_name)
            input_dir = _stage_jdx_files(args.jdx_files, temp_root)
            bundle = load_spectra_dir(input_dir)
            verify_zero_filled_context(bundle)

            image_meta = render_spectra_images(
                input_dir,
                temp_root / "spectra_images",
                max_image_side=args.max_image_side,
            )
            all_images = {k: Path(v) for k, v in image_meta.get("image_files", {}).items()}
            present_images = {k: all_images[k] for k in bundle.present_spectra if k in all_images}

            route_info: dict[str, Any]
            runtime: VLMRuntime | None = None
            if args.skip_vlm_generate:
                generated_text = "<spectra_st>"
                route_info = {
                    "generated_text": generated_text,
                    "has_spectra_token": True,
                    "has_spectra_token_text": True,
                    "image_order": list(present_images),
                    "prompt": question,
                    "skipped": True,
                }
            else:
                with redirect_stdout(sys.stderr):
                    runtime = VLMRuntime(
                        merged_model_dir=merged_model_dir,
                        device=args.device,
                        dtype=args.dtype,
                    )
                    route_info = runtime.generate_route(
                        modality_images=present_images,
                        prompt=question,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                generated_text = route_info["generated_text"]

            should_generate = bool(route_info.get("has_spectra_token")) or bool(args.force_spectra)
            if should_generate:
                if runtime is None:
                    with redirect_stdout(sys.stderr):
                        runtime = VLMRuntime(
                            merged_model_dir=merged_model_dir,
                            device=args.device,
                            dtype=args.dtype,
                        )
                prompts = dict(DEFAULT_PROMPTS)
                if args.ir_prompt:
                    prompts["ir"] = args.ir_prompt
                if args.raman_prompt:
                    prompts["raman"] = args.raman_prompt
                if args.uv_prompt:
                    prompts["uv"] = args.uv_prompt

                spans: dict[str, torch.Tensor] = {}
                for modality in MODALITIES:
                    if modality not in present_images:
                        continue
                    with redirect_stdout(sys.stderr):
                        hidden = runtime.extract_vision_hidden(
                            present_images[modality],
                            modality=modality,
                            prompt=prompts[modality],
                        )
                    if not hidden["ok"]:
                        raise RuntimeError(hidden["error"] or f"{modality} vision hidden extraction failed")
                    spans[modality] = hidden["vision_span"]

                bank = S1OmniLinearPredictorBank.from_state_dicts(
                    modules,
                    device=args.device,
                    dtype=_resolve_dtype(args.dtype),
                )
                motif_out = bank.predict(spans)
                route_metadata = {
                    "has_spectra_token": bool(route_info.get("has_spectra_token")),
                    "image_order": route_info.get("image_order"),
                    "prompt": question,
                    "motif_fusion": motif_out["motif_fusion"],
                    "predictor_modalities_used": motif_out["modalities_used"],
                    "per_modality_prompt": {k: prompts[k] for k in spans},
                    "hf_inline_modules": True,
                }
                with redirect_stdout(sys.stderr):
                    model = Spec2MolModel.from_inline_modules(
                        args.config,
                        modules,
                        device=args.device,
                    )
                    result = model.generate_with_external_motif(
                        bundle=bundle,
                        motif_onehot=motif_out["motif_onehot"],
                        motif_probabilities=motif_out["probabilities"],
                        route_metadata=route_metadata,
                    )
                    if model.save_sdf(result, sdf_path):
                        final_sdf_path = sdf_path

        final_text = _make_final_text(generated_text, final_sdf_path, force_spectra=args.force_spectra)
        print(_terminal_json(question, generated_text, final_text))
        return 0 if (not should_generate or final_sdf_path is not None) else 1
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc(), file=sys.stderr)
        final_text = _make_final_text(generated_text, None, force_spectra=args.force_spectra)
        if not generated_text:
            generated_text = f"[ERROR] {exc}"
            final_text = generated_text
        print(_terminal_json(question, generated_text, final_text))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
