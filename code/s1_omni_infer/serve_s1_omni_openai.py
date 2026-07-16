#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import os
import re
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import infer_s1_omni_checkpoint as dual


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8009
DEFAULT_MODEL_NAME = "s1-omni"
DATA_URL_RE = re.compile(r"^data:(?P<mime>[-\w.]+/[-\w.+]+)?;base64,(?P<data>.+)$", re.DOTALL)
DEBUG_ERRORS = os.getenv("S1_DEBUG_ERRORS", "").lower() in {"1", "true", "yes", "on"}


@dataclass
class RequestContext:
    question: str
    generation_messages: list[dict[str, Any]]
    image_paths: list[str] = field(default_factory=list)
    jdx_files: list[str] = field(default_factory=list)
    pdb_files: list[str] = field(default_factory=list)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None


@dataclass
class RuntimeState:
    args: argparse.Namespace | None = None
    stats: dict[str, Any] | None = None
    model: Any = None
    tokenizer: Any = None
    processor: Any = None
    esm_tokenizer: Any = None
    predictor: Any = None
    image_decoder: Any = None
    simplefold_decoder: Any = None
    spec2mol_decoder: Any = None
    lock: Lock = field(default_factory=Lock)


STATE = RuntimeState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if STATE.args is None:
        load_runtime(parse_args())
    yield


app = FastAPI(title="S1-Omni OpenAI-Compatible Service", lifespan=lifespan)


def error_hint(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return (
            "检查请求中的 image_url、jdx_files、pdb_files、checkpoint_dir 或 stats_path。"
            "相对路径会按当前工作目录和项目根目录解析。"
        )
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "检查当前 Python 环境是否安装了该任务需要的依赖，或相关模块是否在项目路径下。"
    if "out of memory" in message or "cuda" in message:
        return "检查 CUDA 设备、显存占用、S1_DEVICE/S1_IMAGE_DEVICE/S1_SPEC2MOL_DEVICE 等设备配置。"
    if isinstance(exc, ValueError):
        return "检查请求 JSON 的 messages/content 格式、输入模态字段，以及 image_url/jdx_files/pdb_files 是否互斥。"
    if isinstance(exc, RuntimeError):
        return "模型运行阶段失败；检查模型是否加载成功、输入文件是否可读，以及对应模态 decoder 是否可用。"
    return "查看 error.message；如需完整 traceback，可设置环境变量 S1_DEBUG_ERRORS=1 后重启服务。"


def exception_chain(exc: Exception) -> list[dict[str, str]]:
    chain = []
    seen: set[int] = set()
    current: BaseException | None = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": current.__class__.__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return chain


def error_response(exc: Exception, payload: dict[str, Any] | None, path: str) -> dict[str, Any]:
    error: dict[str, Any] = {
        "message": str(exc) or exc.__class__.__name__,
        "type": exc.__class__.__name__,
        "hint": error_hint(exc),
        "request_path": path,
        "request_id": uuid.uuid4().hex,
    }
    causes = exception_chain(exc)
    if causes:
        error["causes"] = causes
    if payload is not None:
        error["request"] = {
            "model": payload.get("model"),
            "stream": bool(payload.get("stream")),
            "fields": sorted(str(key) for key in payload.keys()),
        }
    if DEBUG_ERRORS:
        error["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {"error": error, "detail": error["message"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=os.getenv("S1_CHECKPOINT_DIR", dual.DEFAULT_CHECKPOINT))
    parser.add_argument("--stats_path", default=os.getenv("S1_STATS_PATH", dual.STATS_PATH))
    parser.add_argument("--host", default=os.getenv("S1_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("S1_PORT", DEFAULT_PORT)))
    parser.add_argument("--model_name", default=os.getenv("S1_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--device", default=os.getenv("S1_DEVICE", dual.DEFAULT_MODEL_DEVICE))
    parser.add_argument("--dtype", default=os.getenv("S1_DTYPE", "bf16"), choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default=os.getenv("S1_ATTN_IMPL", "flash_attention_2"))
    parser.add_argument("--max_new_tokens", type=int, default=int(os.getenv("S1_MAX_NEW_TOKENS", "2048")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("S1_TEMPERATURE", "0.2")))
    parser.add_argument("--top_p", type=float, default=float(os.getenv("S1_TOP_P", "0.95")))
    parser.add_argument("--protein_threshold", type=float, default=float(os.getenv("S1_PROTEIN_THRESHOLD", "0.9")))
    parser.add_argument("--protein_output_dir", default=os.getenv("S1_PROTEIN_OUTPUT_DIR", dual.DEFAULT_PROTEIN_OUTPUT_DIR))
    parser.add_argument("--protein_viewer_cdn", default=os.getenv("S1_PROTEIN_VIEWER_CDN", dual.DEFAULT_PROTEIN_VIEWER_CDN))
    parser.add_argument("--protein_summary_max_new_tokens", type=int, default=int(os.getenv("S1_PROTEIN_SUMMARY_MAX_NEW_TOKENS", "2048")))
    parser.add_argument("--strip_protein_for_generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_output_dir", default=os.getenv("S1_IMAGE_OUTPUT_DIR", dual.DEFAULT_IMAGE_OUTPUT_DIR))
    parser.add_argument("--image_height", type=int, default=int(os.getenv("S1_IMAGE_HEIGHT", "1024")))
    parser.add_argument("--image_width", type=int, default=int(os.getenv("S1_IMAGE_WIDTH", "1024")))
    parser.add_argument("--image_steps", type=int, default=int(os.getenv("S1_IMAGE_STEPS", "50")))
    parser.add_argument("--image_edit_steps", type=int, default=int(os.getenv("S1_IMAGE_EDIT_STEPS", "40")))
    parser.add_argument("--image_cfg", type=float, default=float(os.getenv("S1_IMAGE_CFG", "4.0")))
    parser.add_argument("--image_seed", type=int, default=int(os.getenv("S1_IMAGE_SEED", "42")))
    parser.add_argument("--image_device", default=os.getenv("S1_IMAGE_DEVICE", dual.DEFAULT_DECODER_DEVICE))
    parser.add_argument("--image_dtype", default=os.getenv("S1_IMAGE_DTYPE"), choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image_config_name", default=os.getenv("S1_IMAGE_CONFIG_NAME", dual.IMAGE_CONFIG_NAME))
    parser.add_argument("--load_image_decoder_at_start", action="store_true")
    parser.add_argument("--simplefold_output_dir", default=os.getenv("S1_SIMPLEFOLD_OUTPUT_DIR", dual.DEFAULT_SIMPLEFOLD_OUTPUT_DIR))
    parser.add_argument("--simplefold_output_format", default=os.getenv("S1_SIMPLEFOLD_OUTPUT_FORMAT", "cif"), choices=["cif", "pdb"])
    parser.add_argument("--simplefold_device", default=os.getenv("S1_SIMPLEFOLD_DEVICE", dual.DEFAULT_DECODER_DEVICE))
    parser.add_argument("--simplefold_dtype", default=os.getenv("S1_SIMPLEFOLD_DTYPE", "fp32"), choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--simplefold_ccd_cache_dir", default=os.getenv("S1_SIMPLEFOLD_CCD_CACHE", dual.DEFAULT_SIMPLEFOLD_CCD_CACHE))
    parser.add_argument("--simplefold_torch_hub_dir", default=os.getenv("S1_SIMPLEFOLD_TORCH_HUB", dual.DEFAULT_SIMPLEFOLD_TORCH_HUB))
    parser.add_argument("--simplefold_workspace", default=os.getenv("S1_SIMPLEFOLD_WORKSPACE"))
    parser.add_argument("--simplefold_num_steps", type=int, default=int(os.getenv("S1_SIMPLEFOLD_NUM_STEPS", "500")))
    parser.add_argument("--simplefold_tau", type=float, default=float(os.getenv("S1_SIMPLEFOLD_TAU", "0.1")))
    parser.add_argument("--simplefold_seed", type=int, default=int(os.getenv("S1_SIMPLEFOLD_SEED", "42")))
    parser.add_argument("--simplefold_sample_id", default=os.getenv("S1_SIMPLEFOLD_SAMPLE_ID"))
    parser.add_argument("--simplefold_route_token_policy", default=os.getenv("S1_SIMPLEFOLD_ROUTE_TOKEN_POLICY", "last"), choices=["first", "last"])
    parser.add_argument("--load_simplefold_decoder_at_start", action="store_true")
    parser.add_argument("--spectra_output_dir", default=os.getenv("S1_SPECTRA_OUTPUT_DIR", dual.DEFAULT_SPECTRA_OUTPUT_DIR))
    parser.add_argument("--spec2mol_config", default=os.getenv("S1_SPEC2MOL_CONFIG", dual.DEFAULT_SPEC2MOL_CONFIG))
    parser.add_argument("--spec2mol_device", default=os.getenv("S1_SPEC2MOL_DEVICE", dual.DEFAULT_DECODER_DEVICE))
    parser.add_argument("--ir_prompt", default=os.getenv("S1_IR_PROMPT"))
    parser.add_argument("--raman_prompt", default=os.getenv("S1_RAMAN_PROMPT"))
    parser.add_argument("--uv_prompt", default=os.getenv("S1_UV_PROMPT"))
    parser.add_argument("--spectra_max_image_side", type=int, default=None)
    parser.add_argument("--force_spectra", action="store_true")
    parser.add_argument("--load_spec2mol_decoder_at_start", action="store_true")
    parser.add_argument("--load_processor_at_start", action=argparse.BooleanOptionalAction, default=True)
    args, _ = parser.parse_known_args()
    return args


def ensure_args_for_loader(args: argparse.Namespace) -> argparse.Namespace:
    loader_args = copy.copy(args)
    loader_args.question = None
    loader_args.question_file = None
    loader_args.input_file = None
    loader_args.output_file = None
    loader_args.image_input = None
    loader_args.jdx_files = None
    loader_args.pdb_files = None
    return loader_args


def load_runtime(args: argparse.Namespace) -> None:
    loader_args = ensure_args_for_loader(args)
    STATE.args = args
    STATE.stats = dual.load_regression_stats(args.stats_path)
    (
        STATE.model,
        STATE.tokenizer,
        STATE.processor,
        STATE.esm_tokenizer,
        STATE.predictor,
        STATE.image_decoder,
        STATE.simplefold_decoder,
        STATE.spec2mol_decoder,
    ) = dual.load_dual_model(loader_args)
    if args.load_processor_at_start and STATE.processor is None:
        STATE.processor = dual.load_qwen3_vl_processor(args.checkpoint_dir)


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            part_type = part.get("type", "text")
            if part_type == "text":
                parts.append(str(part.get("text", part.get("content", ""))))
        return "\n".join(x for x in parts if x).strip()
    return str(content)


def ensure_request_temp(ctx: RequestContext) -> Path:
    if ctx.temp_dir is None:
        ctx.temp_dir = tempfile.TemporaryDirectory(prefix="s1_omni_api_")
    return Path(ctx.temp_dir.name)


def resolve_local_path(value: str | Path) -> str:
    raw = Path(str(value)).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(dual.project_root / raw)
        candidates.append(dual.project_root.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def stage_data_url(ctx: RequestContext, value: str, suffix: str) -> str:
    match = DATA_URL_RE.match(value)
    if match is None:
        raise ValueError("invalid data URL")
    temp_root = ensure_request_temp(ctx)
    payload = base64.b64decode(match.group("data"))
    path = temp_root / f"upload_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(payload)
    return str(path)


def stage_inline_text(ctx: RequestContext, value: str, suffix: str) -> str:
    temp_root = ensure_request_temp(ctx)
    path = temp_root / f"inline_{uuid.uuid4().hex}{suffix}"
    path.write_text(value, encoding="utf-8")
    return str(path)


def normalize_image_url(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("url") or raw.get("path")
    return str(raw)


def append_jdx(ctx: RequestContext, raw: Any) -> None:
    values = raw if isinstance(raw, list) else [raw]
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            if "jdx_files" in value:
                append_jdx(ctx, value.get("jdx_files"))
                continue
            value = value.get("path") or value.get("url") or value.get("content")
        text = str(value)
        if text.startswith("data:"):
            ctx.jdx_files.append(stage_data_url(ctx, text, ".jdx"))
        elif "\n" in text or text.lstrip().startswith("##"):
            ctx.jdx_files.append(stage_inline_text(ctx, text, ".jdx"))
        else:
            ctx.jdx_files.append(resolve_local_path(text))


def append_pdb(ctx: RequestContext, raw: Any) -> None:
    values = raw if isinstance(raw, list) else [raw]
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            if "pdb_files" in value:
                append_pdb(ctx, value.get("pdb_files"))
                continue
            value = value.get("path") or value.get("url") or value.get("content")
        text = str(value)
        if text.startswith("data:"):
            ctx.pdb_files.append(stage_data_url(ctx, text, ".pdb"))
        elif "\n" in text or text.lstrip().startswith(("ATOM", "HETATM", "HEADER", "data_")):
            suffix = ".cif" if text.lstrip().startswith("data_") else ".pdb"
            ctx.pdb_files.append(stage_inline_text(ctx, text, suffix))
        else:
            ctx.pdb_files.append(resolve_local_path(text))


def normalize_messages(payload: dict[str, Any], ctx: RequestContext) -> tuple[list[dict[str, Any]], str]:
    raw_messages = payload.get("messages")
    if raw_messages is None:
        question = str(payload.get("question", payload.get("prompt", ""))).strip()
        if not question:
            raise ValueError("request must contain messages, question, or prompt")
        return [{"role": "user", "content": [{"type": "text", "text": question}]}], question
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be a list")

    system_prompt = payload.get("system_prompt")
    normalized: list[dict[str, Any]] = []
    latest_user_text = ""

    if system_prompt:
        normalized.append({"role": "system", "content": str(system_prompt)})

    flat_content = all(isinstance(item, dict) and "role" not in item for item in raw_messages)
    source_messages = [{"role": "user", "content": raw_messages}] if flat_content else raw_messages

    for message in source_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role == "system":
            text = text_from_content(content)
            if text:
                normalized.append({"role": "system", "content": text})
            continue

        parts: list[dict[str, Any]] = []
        content_items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for part in content_items:
            if not isinstance(part, dict):
                parts.append({"type": "text", "text": str(part)})
                continue
            part_type = str(part.get("type", "text"))
            if part_type == "text":
                text = str(part.get("text", part.get("content", "")))
                if text:
                    parts.append({"type": "text", "text": text})
            elif part_type in {"system", "system_prompt"}:
                text = str(part.get("text", part.get("content", ""))).strip()
                if text:
                    normalized.append({"role": "system", "content": text})
            elif part_type == "image_url":
                url = normalize_image_url(part.get("image_url", part.get("url")))
                if not url:
                    continue
                if url.startswith("data:"):
                    path = stage_data_url(ctx, url, ".png")
                else:
                    path = resolve_local_path(url)
                ctx.image_paths.append(path)
                parts.append({"type": "image", "image": path})
            elif part_type in {"image", "input_image"}:
                url = normalize_image_url(part.get("image", part.get("path", part.get("url"))))
                if not url:
                    continue
                if url.startswith("data:"):
                    path = stage_data_url(ctx, url, ".png")
                else:
                    path = resolve_local_path(url)
                ctx.image_paths.append(path)
                parts.append({"type": "image", "image": path})
            elif part_type in {"jdx_files", "jdx_file", "jdx"}:
                append_jdx(ctx, part.get("jdx_files", part.get("content", part.get("path"))))
            elif part_type in {"pdb_files", "pdb_file", "pdb"}:
                append_pdb(ctx, part.get("pdb_files", part.get("content", part.get("path"))))

        text = text_from_content(parts)
        if role == "user" and text:
            latest_user_text = text
        if parts or role == "assistant":
            normalized.append({"role": role, "content": parts if parts else text_from_content(content)})

    if not normalized:
        raise ValueError("messages did not contain usable content")
    if not latest_user_text:
        latest_user_text = "\n".join(
            text_from_content(message.get("content"))
            for message in normalized
            if message.get("role") != "system"
        ).strip()
    if not latest_user_text:
        latest_user_text = "请根据输入内容完成任务。"
    return normalized, latest_user_text


def load_images_for_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    images = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                image = dual._load_rgb_image(part["image"])
                images.append(image)
                new_content.append({"type": "image", "image": image})
            else:
                new_content.append(part)
        converted.append({"role": message.get("role", "user"), "content": new_content})
    return converted, images


def apply_request_overrides(args: argparse.Namespace, payload: dict[str, Any], ctx: RequestContext) -> argparse.Namespace:
    request_args = copy.copy(args)
    request_args.image_input = ctx.image_paths[0] if ctx.image_paths else None
    request_args.image_inputs = list(ctx.image_paths)
    request_args.jdx_files = list(ctx.jdx_files) or None
    request_args.pdb_files = list(ctx.pdb_files) or None
    for source_key, attr, caster in (
        ("max_tokens", "max_new_tokens", int),
        ("max_new_tokens", "max_new_tokens", int),
        ("temperature", "temperature", float),
        ("top_p", "top_p", float),
        ("seed", "image_seed", int),
    ):
        if source_key in payload and payload[source_key] is not None:
            setattr(request_args, attr, caster(payload[source_key]))
    return request_args


def make_context(payload: dict[str, Any]) -> RequestContext:
    ctx = RequestContext(question="", generation_messages=[])
    try:
        if payload.get("pdb_files"):
            append_pdb(ctx, payload.get("pdb_files"))
        messages, question = normalize_messages(payload, ctx)
        ctx.question = question
        ctx.generation_messages = messages
        return ctx
    except Exception:
        ctx.cleanup()
        raise


def tokenizer_generation_inputs(tokenizer, messages: list[dict[str, Any]], device: str) -> dict[str, Any]:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def image_generation_inputs(processor, messages: list[dict[str, Any]], device: str) -> dict[str, Any]:
    if processor is None:
        raise RuntimeError("Qwen3VLProcessor is required for image requests")
    image_messages, images = load_images_for_messages(messages)
    if not images:
        raise RuntimeError("image request did not contain readable images")
    text = dual._apply_processor_template(processor, image_messages)
    inputs = processor(
        text=[text],
        images=[dual.resize_for_encoder(image) for image in images],
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def generate_route(
    model,
    tokenizer,
    processor,
    question: str,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    spectra_context: dict[str, Any] | None = None,
):
    generation_question = dual.prepare_generation_question(
        question,
        use_esm2=getattr(model, "use_esm2", False),
        strip_protein=args.strip_protein_for_generation,
    )
    if spectra_context is not None:
        inputs = dual.prepare_spectra_generation_inputs(
            processor,
            generation_question,
            spectra_context,
            args.device,
        )
    elif getattr(args, "image_inputs", None):
        inputs = image_generation_inputs(processor, messages, args.device)
    else:
        inputs = tokenizer_generation_inputs(tokenizer, messages, args.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p if args.temperature > 0 else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    generated_ids = model.backbone.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    user_spans = dual._find_chat_role_token_spans(
        tokenizer,
        inputs["input_ids"],
        inputs.get("attention_mask"),
        role="user",
    )
    condition_drop_idx = int(user_spans[0][0]) if user_spans and user_spans[0] is not None else prompt_len
    new_ids = generated_ids[0, prompt_len:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=False).strip()
    route_token = dual.extract_final_route_token(tokenizer, new_ids, generated_text)
    return generation_question, generated_ids, generated_text, route_token, condition_drop_idx


def infer_with_context(payload: dict[str, Any]) -> dict[str, Any]:
    if STATE.args is None:
        raise RuntimeError("runtime is not loaded")
    ctx = make_context(payload)
    args = apply_request_overrides(STATE.args, payload, ctx)
    try:
        if args.image_input and args.jdx_files:
            raise ValueError("image_url and jdx_files cannot be used in the same request")
        if args.pdb_files and (args.image_input or args.jdx_files):
            raise ValueError("pdb_files cannot be used with image_url or jdx_files in the same request")

        with STATE.lock:
            temp_dir: tempfile.TemporaryDirectory[str] | None = None
            try:
                if args.pdb_files:
                    prediction = dual.infer_protein_from_pdb(
                        STATE.model,
                        STATE.tokenizer,
                        STATE.processor,
                        STATE.esm_tokenizer,
                        ctx.question,
                        args,
                    )
                    return {
                        "question": ctx.question,
                        "generation_question": prediction["generation_questions"],
                        "llm_output": prediction["llm_outputs"],
                        "final_special_token": prediction["final_special_token"],
                        "selected_by": "protein",
                        "prediction": dual.to_jsonable(prediction),
                        "final_text": prediction["answer_text"],
                    }

                spectra_context = None
                if args.jdx_files:
                    temp_dir = tempfile.TemporaryDirectory(prefix="s1_omni_spectra_")
                    spectra_context = dual.prepare_spectra_context(args, Path(temp_dir.name))

                with torch.inference_mode():
                    generation_question, generated_ids, generated_text, route_token, condition_drop_idx = generate_route(
                        STATE.model,
                        STATE.tokenizer,
                        STATE.processor,
                        ctx.question,
                        ctx.generation_messages,
                        args,
                        spectra_context=spectra_context,
                    )

                prediction: dict[str, Any] | None = None
                replacement = ""
                selected_by = "unknown"
                if route_token == dual.PROT_CLA_TOKEN:
                    selected_by = "protein"
                    prediction = dual.infer_protein(STATE.model, STATE.tokenizer, STATE.esm_tokenizer, ctx.question, args)
                    replacement = json.dumps(prediction["positive_indices"], ensure_ascii=False)
                elif route_token == dual.PROTEIN_FOLD_TOKEN:
                    selected_by = "simplefold"
                    prediction = dual.infer_simplefold(
                        STATE.model,
                        STATE.tokenizer,
                        STATE.simplefold_decoder,
                        generated_ids,
                        ctx.question,
                        args,
                    )
                    replacement = prediction["structure_path"]
                elif route_token in {dual.LINEAR_CLA_TOKEN, dual.LINEAR_PRE_TOKEN}:
                    prediction = dual.infer_property(STATE.model, STATE.tokenizer, STATE.predictor, generated_ids, STATE.stats, args)
                    if route_token == dual.LINEAR_CLA_TOKEN:
                        selected_by = "classification"
                        replacement = str(prediction["classification"]["pred_class"])
                        if replacement == "0":
                            replacement = "No"
                        else:
                            replacement = "Yes"
                    else:
                        selected_by = "regression"
                        replacement = str(prediction["regression"]["value"])
                elif route_token in {dual.IMAGE_GEN_TOKEN, dual.IMAGE_EDIT_TOKEN}:
                    source_image = dual.load_pil_image(args.image_input)
                    decoder = STATE.image_decoder.get()
                    image, image_task = decoder.infer_from_generated(
                        route_token=route_token,
                        generated_ids=generated_ids,
                        condition_drop_idx=condition_drop_idx,
                        source_image=source_image,
                        args=args,
                    )
                    image_path = dual.save_generated_image(image, args.image_output_dir, None, image_task)
                    selected_by = image_task
                    prediction = {
                        "type": "image",
                        "task": image_task,
                        "image_path": image_path,
                        "height": args.image_height,
                        "width": args.image_width,
                        "steps": args.image_edit_steps if image_task == "image_edit" else args.image_steps,
                        "cfg": None if image_task == "image_edit" else args.image_cfg,
                        "seed": args.image_seed,
                        "source_image": args.image_input if image_task == "image_edit" else None,
                    }
                    replacement = image_path
                elif route_token == dual.SPECTRA_TOKEN or args.force_spectra:
                    if spectra_context is None:
                        raise ValueError("jdx_files is required when the selected route is <spectra_st>")
                    selected_by = "spectra"
                    prediction = dual.infer_spectra(
                        STATE.model,
                        STATE.processor,
                        STATE.spec2mol_decoder,
                        spectra_context,
                        ctx.question,
                        args,
                    )
                    replacement = prediction.get("sdf_path") or prediction.get("smiles") or ""

                if prediction is None:
                    final_text = generated_text
                elif selected_by == "spectra" and args.force_spectra and route_token != dual.SPECTRA_TOKEN and dual.SPECTRA_TOKEN not in generated_text:
                    final_text = replacement
                else:
                    final_text = dual.replace_last_route_token(generated_text, route_token, replacement)

                return {
                    "question": ctx.question,
                    "generation_question": generation_question,
                    "llm_output": generated_text,
                    "final_special_token": route_token,
                    "selected_by": selected_by,
                    "prediction": dual.to_jsonable(prediction),
                    "final_text": final_text,
                }
            finally:
                if temp_dir is not None:
                    temp_dir.cleanup()
    finally:
        ctx.cleanup()


def openai_completion(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    created = int(time.time())
    model_name = str(payload.get("model") or (STATE.args.model_name if STATE.args else DEFAULT_MODEL_NAME))
    final_text = str(result.get("final_text", ""))
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": "stop",
            }
        ],
        "final_text": final_text,
    }
    if payload.get("return_full_result", True):
        response["result"] = result
    return response


def sse_event(data: Any) -> str:
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


async def stream_completion(payload: dict[str, Any], path: str = "/v1/chat/completions"):
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, infer_with_context, payload)
        final_text = str(result.get("final_text", ""))
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = str(payload.get("model") or (STATE.args.model_name if STATE.args else DEFAULT_MODEL_NAME))
        chunk_size = int(payload.get("stream_chunk_size") or 64)
        for start in range(0, len(final_text), chunk_size):
            chunk = final_text[start : start + chunk_size]
            yield sse_event(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk, "final_text": chunk},
                            "finish_reason": None,
                        }
                    ],
                    "final_text": chunk,
                }
            )
        yield sse_event(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "final_text": "",
            }
        )
        yield sse_event("[DONE]")
    except Exception as exc:  # noqa: BLE001
        yield sse_event(error_response(exc, payload, path))
        yield sse_event("[DONE]")


async def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, infer_with_context, payload)


async def read_json_payload(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    path = str(request.url.path)
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return None, JSONResponse(error_response(exc, None, path), status_code=400)
    if not isinstance(payload, dict):
        exc = ValueError("request JSON body must be an object")
        return None, JSONResponse(error_response(exc, None, path), status_code=400)
    return payload, None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if STATE.model is not None else "loading",
        "model": STATE.args.model_name if STATE.args else DEFAULT_MODEL_NAME,
        "device": STATE.args.device if STATE.args else None,
    }


@app.post("/predict")
async def predict(request: Request):
    payload, error = await read_json_payload(request)
    if error is not None:
        return error
    if payload.get("stream"):
        return StreamingResponse(stream_completion(payload, str(request.url.path)), media_type="text/event-stream")
    try:
        return JSONResponse(await run_payload(payload))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(error_response(exc, payload, str(request.url.path)), status_code=400)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload, error = await read_json_payload(request)
    if error is not None:
        return error
    if payload.get("stream"):
        return StreamingResponse(stream_completion(payload, str(request.url.path)), media_type="text/event-stream")
    try:
        result = await run_payload(payload)
        return JSONResponse(openai_completion(payload, result))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(error_response(exc, payload, str(request.url.path)), status_code=400)


@app.post("/chat/completions")
async def chat_completions_alias(request: Request):
    return await chat_completions(request)


def main() -> None:
    args = parse_args()
    load_runtime(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
