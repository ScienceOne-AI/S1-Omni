"""Runtime wrapper around the merged Qwen3-VL model.

Loads the merged HuggingFace directory (the SFT'd 32B VLM + spec2mol merge
metadata) and provides two entry points:

  * ``generate_route(ir_png, raman_png, uv_png, prompt)`` — three-image VLM
    generate that decides whether to emit the ``<spectra_st>`` route token.
  * ``extract_ir_vision_hidden(ir_png, prompt)`` — single IR-image forward pass
    returning the ``[1, 546, 5120]`` vision-span hidden state used by the
    modality-specific S1-Omni linear predictor.

The IR image is resized to the training reference dimensions (1024x552) so the
processor yields ``image_grid_thw=[1,34,64]`` -> 544 image_pad tokens -> a
contiguous vision span of length 546, matching the E1 extractor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from spec2mol_v2.s1_omni_linear_predictor import DEFAULT_PROMPTS
from spec2mol_v2.vision_span import (
    EXPECTED_VISION_SPAN,
    VISION_START_ID,
    slice_vision_span,
)

# Training recipe: 1024x552 IR PNG + max_pixels=688128 -> grid [1,34,64] -> 546 span.
DEFAULT_MAX_PIXELS = 688128
IMAGE_TARGET_SIZE = (1024, 552)  # (width, height)
IR_TARGET_SIZE = IMAGE_TARGET_SIZE

ROUTE_DEFAULT_PROMPT = "请你详细分析这张谱图，预测可能的分子结构"
IR_HIDDEN_DEFAULT_PROMPT = DEFAULT_PROMPTS["ir"]


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.bfloat16
    if isinstance(dtype, torch.dtype):
        return dtype
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return table.get(dtype.lower(), torch.bfloat16)


def _load_model(merged_model_dir: Path, dtype: torch.dtype, attn_implementation: str | None):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        str(merged_model_dir), trust_remote_code=True, max_pixels=DEFAULT_MAX_PIXELS
    )
    kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(merged_model_dir), dtype=dtype, **kwargs
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(merged_model_dir), torch_dtype=dtype, **kwargs
        )
    model.eval()
    return model, processor


def _load_image(path: str | Path, target_size: tuple[int, int] | None = None) -> Image.Image:
    with Image.open(path) as im:
        image = im.convert("RGB").copy()
    if target_size is not None and image.size != target_size:
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        image = image.resize(target_size, resample=resample)
    return image


def _build_messages(images: list[Image.Image], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _apply_template(processor, messages: list[dict[str, Any]]) -> str:
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


class VLMRuntime:
    """Loads the merged HF model once and serves route + IR-hidden requests."""

    def __init__(
        self,
        merged_model_dir: str | Path,
        device: str = "cuda:0",
        dtype: str | torch.dtype | None = "bf16",
        attn_implementation: str | None = None,
        max_pixels: int | None = None,
    ):
        self.merged_model_dir = str(Path(merged_model_dir).expanduser().resolve())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = _resolve_dtype(dtype)
        # max_pixels is fixed at load via the processor; record for metadata.
        self.max_pixels = max_pixels if max_pixels is not None else DEFAULT_MAX_PIXELS

        self.model, self.processor = _load_model(
            Path(self.merged_model_dir), self.torch_dtype, attn_implementation
        )
        self.model.to(self.device)
        self.tokenizer = self.processor.tokenizer
        self.spectra_token_id = self.tokenizer.convert_tokens_to_ids("<spectra_st>")

    # -- public API ----------------------------------------------------------

    @torch.no_grad()
    def generate_route(
        self,
        ir_png: str | Path | None = None,
        raman_png: str | Path | None = None,
        uv_png: str | Path | None = None,
        prompt: str = ROUTE_DEFAULT_PROMPT,
        max_new_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.95,
        modality_images: dict[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        prompt = prompt or ROUTE_DEFAULT_PROMPT
        image_map = modality_images or {
            k: v for k, v in {"ir": ir_png, "raman": raman_png, "uv": uv_png}.items() if v is not None
        }
        image_order = [k for k in ("ir", "raman", "uv") if k in image_map]
        if not image_order:
            raise ValueError("generate_route requires at least one modality image")
        images = [_load_image(image_map[k]) for k in image_order]
        messages = _build_messages(images, prompt)
        text = _apply_template(self.processor, messages)
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=False
        ).to(self.device)

        input_len = int(inputs["input_ids"].shape[1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "temperature": temperature,
        }
        if temperature <= 0:
            gen_kwargs.pop("top_p")
            gen_kwargs.pop("temperature")
            gen_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0, input_len:]
        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)

        has_spectra_token = "<spectra_st>" in generated_text
        has_spectra_token_id = False
        if self.spectra_token_id is not None and self.spectra_token_id != self.tokenizer.unk_token_id:
            has_spectra_token_id = int(self.spectra_token_id) in new_ids.detach().cpu().tolist()

        return {
            "generated_text": generated_text,
            "has_spectra_token": bool(has_spectra_token or has_spectra_token_id),
            "has_spectra_token_text": bool(has_spectra_token),
            "has_spectra_token_id": bool(has_spectra_token_id),
            "spectra_token_id": int(self.spectra_token_id) if self.spectra_token_id is not None else None,
            "image_order": image_order,
            "prompt": prompt,
            "input_length": input_len,
            "new_token_count": int(new_ids.shape[0]),
            "new_token_ids": new_ids.detach().cpu().tolist(),
        }

    @torch.no_grad()
    def extract_vision_hidden(
        self,
        image_path: str | Path,
        modality: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Single-modality image forward pass -> ``[1, 546, 5120]`` vision-span hidden."""
        if modality not in DEFAULT_PROMPTS:
            raise ValueError(f"unsupported modality: {modality}")
        prompt = prompt or DEFAULT_PROMPTS[modality]
        image = _load_image(image_path, target_size=IMAGE_TARGET_SIZE)
        messages = _build_messages([image], prompt)
        text = _apply_template(self.processor, messages)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=False
        ).to(self.device)

        input_ids = inputs["input_ids"]
        image_grid_thw = inputs.get("image_grid_thw")
        base = self.model.model if hasattr(self.model, "model") else self.model
        # transformers 5.6.x Qwen3VL requires mm_token_type_ids (returned by the
        # processor alongside input_ids) for M-RoPE. Forward every model-relevant
        # field the processor produced so we don't drop anything it expects.
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "use_cache": False,
            "return_dict": True,
        }
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw
        for optional_key in ("mm_token_type_ids", "video_grid_thw", "position_ids"):
            if optional_key in inputs:
                forward_kwargs[optional_key] = inputs[optional_key]
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
            "modality": modality,
            "prompt": prompt,
            "vision_span": vision_span.detach().cpu() if vision_span is not None else None,
            "vision_span_shape": (tuple(vision_span.shape) if vision_span is not None else None),
            "ir_vision_span": vision_span.detach().cpu() if vision_span is not None else None,
            "ir_vision_span_shape": (tuple(vision_span.shape) if vision_span is not None else None),
            "input_ids_shape": tuple(input_ids.shape),
            "image_grid_thw": (
                image_grid_thw.detach().cpu().tolist() if image_grid_thw is not None else None
            ),
            "hidden_shape": tuple(hidden.shape),
            "vision_start_id": VISION_START_ID,
            "expected_vision_span": EXPECTED_VISION_SPAN,
            "target_size": list(IMAGE_TARGET_SIZE),
            "ir_target_size": list(IR_TARGET_SIZE),
            "error": err,
        }


# Backward-compatible alias for old callers.
def _extract_ir_compat(self, ir_png: str | Path, prompt: str = IR_HIDDEN_DEFAULT_PROMPT) -> dict[str, Any]:
    return self.extract_vision_hidden(ir_png, modality="ir", prompt=prompt)

VLMRuntime.extract_ir_vision_hidden = _extract_ir_compat
