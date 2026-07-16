"""Loaders for the fused S1-VL + spec2mol small-modules directory.

The fused directory (produced by ``scripts/fuse_vlm_spectra_safetensors.py``)
contains a single ``fused.safetensors`` carrying:

  * VLM tensors under their original HuggingFace names (loaded by transformers
    via ``model.safetensors.index.json``).
  * Four small modules under prefixed keys (``spectra_omni.*`` / ``atomcount.*``
    / ``motif.*`` / ``dmt.*``), loaded here from the unified
    ``model.safetensors.index.json`` by prefix.

This module reconstructs the four small-module model objects from
the self-contained fused index metadata + the indexed tensors, and wraps the VLM with the
same route / IR-hidden extraction logic used in ``vlm_runtime.py``.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from spec2mol_v2.spectra_omni_predictor import (
    EXPECTED_VISION_SPAN,
    VISION_START_ID,
    slice_vision_span,
)

# ---------------------------------------------------------------------------
# transformers 5.6.2 compatibility patch (copied from infer_s1_vl_simplefold_checkpoint.py)
# ---------------------------------------------------------------------------

def patch_transformers_compat() -> None:
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return

    original = getattr(PreTrainedTokenizerBase, "_set_model_specific_special_tokens", None)
    if original is not None and not getattr(original, "_s1_spec2mol_patched", False):

        def patched(self, special_tokens):  # type: ignore[no-untyped-def]
            if isinstance(special_tokens, list):
                if special_tokens:
                    self.add_special_tokens(
                        {"additional_special_tokens": special_tokens},
                        replace_additional_special_tokens=False,
                    )
                return None
            return original(self, special_tokens)

        patched._s1_spec2mol_patched = True  # type: ignore[attr-defined]
        PreTrainedTokenizerBase._set_model_specific_special_tokens = patched

    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLConfig,
            Qwen3VLTextConfig,
        )
    except Exception:
        return

    text_init = getattr(Qwen3VLTextConfig, "__init__", None)
    if text_init is not None and not getattr(text_init, "_s1_spec2mol_patched", False):

        def patched_text_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            text_init(self, *args, **kwargs)
            if getattr(self, "rope_scaling", None) is None and getattr(self, "rope_parameters", None) is not None:
                self.rope_scaling = dict(self.rope_parameters)

        patched_text_init._s1_spec2mol_patched = True  # type: ignore[attr-defined]
        Qwen3VLTextConfig.__init__ = patched_text_init

    config_init = getattr(Qwen3VLConfig, "__init__", None)
    if config_init is not None and not getattr(config_init, "_s1_spec2mol_patched", False):

        def patched_config_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            config_init(self, *args, **kwargs)
            text_config = getattr(self, "text_config", None)
            if (
                text_config is not None
                and getattr(text_config, "rope_scaling", None) is None
                and getattr(text_config, "rope_parameters", None) is not None
            ):
                text_config.rope_scaling = dict(text_config.rope_parameters)

        patched_config_init._s1_spec2mol_patched = True  # type: ignore[attr-defined]
        Qwen3VLConfig.__init__ = patched_config_init


# ---------------------------------------------------------------------------
# small-module loading
# ---------------------------------------------------------------------------

def load_small_modules_config_with_source(fused_dir: str | Path) -> tuple[dict[str, Any], str]:
    fused_dir = Path(fused_dir)
    index_path = fused_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        embedded = index.get("metadata", {}).get("small_modules_config_json")
        if embedded:
            return json.loads(embedded), "model_index_metadata"

    # Backward compatibility for older fused directories. New fusions do not
    # write this file.
    legacy_path = fused_dir / "small_modules_config.json"
    if legacy_path.is_file():
        return json.loads(legacy_path.read_text(encoding="utf-8")), "legacy_small_modules_config_json"
    raise FileNotFoundError(
        f"missing small_modules_config_json in {index_path} and missing {legacy_path}"
    )


def load_small_modules_config(fused_dir: str | Path) -> dict[str, Any]:
    cfg, _ = load_small_modules_config_with_source(fused_dir)
    return cfg


def _load_safetensors_from_weight_map(
    index_path: Path, weight_map: dict[str, str], prefix: str
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    for filename in sorted(set(weight_map.values())):
        path = index_path.parent / filename
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key, mapped in weight_map.items():
                if mapped != filename:
                    continue
                if key not in available:
                    raise KeyError(f"{key!r} missing from {path}")
                out_key = key.removeprefix(prefix) if prefix else key
                state_dict[out_key] = handle.get_tensor(key)
    return state_dict


def load_indexed_safetensors(index_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    """Load tensors listed in an index, stripping ``prefix``."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    return _load_safetensors_from_weight_map(index_path, weight_map, prefix)


def load_prefixed_safetensors(
    fused_dir: Path, prefix: str, legacy_index_name: str | None = None
) -> dict[str, torch.Tensor]:
    """Load tensors from the unified HF index, with legacy per-module fallback."""
    unified_index_path = fused_dir / "model.safetensors.index.json"
    if unified_index_path.is_file():
        index = json.loads(unified_index_path.read_text(encoding="utf-8"))
        weight_map = {
            key: filename
            for key, filename in index.get("weight_map", {}).items()
            if key.startswith(prefix)
        }
        if weight_map:
            return _load_safetensors_from_weight_map(unified_index_path, weight_map, prefix)

    if legacy_index_name is not None:
        legacy_index_path = fused_dir / legacy_index_name
        if legacy_index_path.is_file():
            return load_indexed_safetensors(legacy_index_path, prefix)

    raise FileNotFoundError(
        f"no tensors with prefix {prefix!r} found in {unified_index_path}"
    )


def load_spectra_omni(fused_dir: str | Path, cfg: dict[str, Any], device: str | torch.device) -> Any:
    """Build PrefixAttnExtractor and load spectra_omni.* weights from fused."""
    fused_dir = Path(fused_dir)
    from spec2mol_v2.prefix_attn_extractor import PrefixAttnExtractor

    b = cfg["build_args"]
    model = PrefixAttnExtractor(
        in_dim=b["in_dim"], proj_dim=b["proj_dim"], n_layers=b["n_layers"], n_heads=b["n_heads"],
        ffn_mult=b["ffn_mult"], dropout=b["dropout"], num_queries=b["num_queries"],
        num_classes=b["num_classes"], feat_dim=b["feat_dim"], frozen_head_state=None,
    )
    prefix = cfg["prefix"]
    state_dict = load_prefixed_safetensors(fused_dir, prefix, "spectra_omni.safetensors.index.json")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device)
    model.eval()
    return model


def load_head(fused_dir: str | Path, name: str, cfg: dict[str, Any], device: str | torch.device, out_dim: int) -> Any:
    fused_dir = Path(fused_dir)
    from spec2mol.heads import MLPHead

    b = cfg["build_args"]
    model = MLPHead(b["in_dim"], out_dim, hidden=b["hidden"], dropout=b["dropout"])
    prefix = cfg["prefix"]
    state_dict = load_prefixed_safetensors(fused_dir, prefix, f"{name}.safetensors.index.json")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device)
    model.eval()
    return model


def load_dmt(fused_dir: str | Path, cfg: dict[str, Any], device: str | torch.device) -> tuple[Any, Any]:
    """Build vendored LLM_DMT model + load dmt.* weights (keeps module. prefix)."""
    fused_dir = Path(fused_dir)

    from spec2mol_v2.dmt_config import get_config
    from spec2mol_v2.llm_dmt_runtime.models import create_model

    config = get_config(device=device)
    config.training.distributed = False
    config.training.num_gpus = 1
    config.training.world_size = 1
    config.training.local_rank = 0
    config.device = torch.device(device)
    config.data.spectra_version = "allspectra"
    # create_model prints "Train SpecFormer from scratch" when the cond_encoder is
    # randomly initialised (the DMT config has no pretrained specformer ckpt). The
    # random init is fully overwritten by the strict load_state_dict below, so the
    # message is noise — swallow it during construction.
    with contextlib.redirect_stdout(io.StringIO()):
        model = create_model(config)
    prefix = cfg["prefix"]
    state_dict = load_prefixed_safetensors(fused_dir, prefix, "dmt.safetensors.index.json")
    # state_dict retains inner "module." prefix; create_model expects it.
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device)
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# VLM loading + route / IR hidden (mirrors vlm_runtime.VLMRuntime but reads fused)
# ---------------------------------------------------------------------------

DEFAULT_MAX_PIXELS = 688128
IR_TARGET_SIZE = (1024, 552)  # (width, height) -> image_grid_thw=[1,34,64] -> 546 span
ROUTE_DEFAULT_PROMPT = "请你基于上述给出的图片，预测该分子可能的结构"
IR_HIDDEN_DEFAULT_PROMPT = "请你基于给出的红外谱,预测该分子可能的"


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.bfloat16
    if isinstance(dtype, torch.dtype):
        return dtype
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32}.get(dtype.lower(), torch.bfloat16)


def _load_image(path: str | Path, target_size: tuple[int, int] | None = None):
    from PIL import Image
    with Image.open(path) as im:
        image = im.convert("RGB").copy()
    if target_size is not None and image.size != target_size:
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        image = image.resize(target_size, resample=resample)
    return image


def _build_messages(images, prompt: str) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _apply_template(processor, messages) -> str:
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class FusedVLMRuntime:
    """VLM loaded from the fused directory + route / IR-hidden helpers."""

    def __init__(self, fused_dir: str | Path, device: str = "cuda:0",
                 dtype: str | torch.dtype | None = "bf16",
                 attn_implementation: str | None = None):
        patch_transformers_compat()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.fused_dir = str(Path(fused_dir).expanduser().resolve())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = _resolve_dtype(dtype)

        self.processor = AutoProcessor.from_pretrained(
            self.fused_dir, trust_remote_code=True, max_pixels=DEFAULT_MAX_PIXELS
        )
        kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        # The fused.safetensors carries 505 small-module tensors alongside the
        # 1058 VLM tensors. transformers' LOAD REPORT prints every small-module
        # key as UNEXPECTED. They are genuinely unused by the VLM (loaded
        # separately via per-module indices), so silence the report during load.
        # transformers passes modeling_utils' module logger into log_state_dict_report.
        silenced_loggers = [
            logging.getLogger("transformers.modeling_utils"),
            logging.getLogger("transformers.utils.loading_report"),
        ]
        orig_levels = [lg.level for lg in silenced_loggers]
        for lg in silenced_loggers:
            lg.setLevel(logging.ERROR)
        try:
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.fused_dir, dtype=self.torch_dtype, **kwargs
                )
            except TypeError:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.fused_dir, torch_dtype=self.torch_dtype, **kwargs
                )
        finally:
            for lg, lvl in zip(silenced_loggers, orig_levels):
                lg.setLevel(lvl)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = self.processor.tokenizer
        self.spectra_token_id = self.tokenizer.convert_tokens_to_ids("<spectra_st>")

    @torch.no_grad()
    def generate_route(self, ir_png, raman_png, uv_png, prompt: str = ROUTE_DEFAULT_PROMPT,
                       max_new_tokens: int = 4096, temperature: float = 0.2, top_p: float = 0.95) -> dict[str, Any]:
        images = [_load_image(ir_png), _load_image(raman_png), _load_image(uv_png)]
        text = _apply_template(self.processor, _build_messages(images, prompt))
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=False).to(self.device)
        input_len = int(inputs["input_ids"].shape[1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
            "top_p": top_p, "temperature": temperature,
        }
        if temperature <= 0:
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0, input_len:]
        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        has_text = "<spectra_st>" in generated_text
        has_id = (self.spectra_token_id is not None
                  and self.spectra_token_id != self.tokenizer.unk_token_id
                  and int(self.spectra_token_id) in new_ids.detach().cpu().tolist())
        return {
            "generated_text": generated_text,
            "has_spectra_token": bool(has_text or has_id),
            "has_spectra_token_text": bool(has_text),
            "has_spectra_token_id": bool(has_id),
            "spectra_token_id": int(self.spectra_token_id) if self.spectra_token_id is not None else None,
            "image_order": ["ir", "raman", "uv"],
            "prompt": prompt,
            "input_length": input_len,
            "new_token_count": int(new_ids.shape[0]),
        }

    @torch.no_grad()
    def extract_ir_vision_hidden(self, ir_png, prompt: str = IR_HIDDEN_DEFAULT_PROMPT) -> dict[str, Any]:
        image = _load_image(ir_png, target_size=IR_TARGET_SIZE)
        text = _apply_template(self.processor, _build_messages([image], prompt))
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=False).to(self.device)
        input_ids = inputs["input_ids"]
        image_grid_thw = inputs.get("image_grid_thw")
        base = self.model.model if hasattr(self.model, "model") else self.model
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "use_cache": False,
            "return_dict": True,
        }
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw
        for opt in ("mm_token_type_ids", "video_grid_thw", "position_ids"):
            if opt in inputs:
                forward_kwargs[opt] = inputs[opt]
        outputs = base(**forward_kwargs)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        try:
            vision_span = slice_vision_span(hidden, input_ids)
            ok = int(vision_span.shape[1]) == EXPECTED_VISION_SPAN
            err = None if ok else f"vision span length {vision_span.shape[1]} != {EXPECTED_VISION_SPAN}"
        except Exception as exc:  # noqa: BLE001
            vision_span = None
            ok = False
            err = repr(exc)
        return {
            "ok": ok,
            "ir_vision_span": vision_span.detach().cpu() if vision_span is not None else None,
            "ir_vision_span_shape": tuple(vision_span.shape) if vision_span is not None else None,
            "input_ids_shape": tuple(input_ids.shape),
            "image_grid_thw": image_grid_thw.detach().cpu().tolist() if image_grid_thw is not None else None,
            "hidden_shape": tuple(hidden.shape),
            "error": err,
        }
