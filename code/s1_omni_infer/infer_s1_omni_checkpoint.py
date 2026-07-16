#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN_ROOT = Path(__file__).resolve().parent
project_root = PROJECT_ROOT
sys.path.append(str(QWEN_ROOT))


def project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def qwen_path(*parts: str) -> str:
    return str(QWEN_ROOT.joinpath(*parts))

from s1_omni.data.data_processor import (  # noqa: E402
    _extract_protein_sequence_and_qwen_text,
    _space_protein_sequence,
)
from s1_omni.modeling_s1_protein import S1Protein  # noqa: E402


DEFAULT_CHECKPOINT = "checkpoints/S1-Omni"
DEFAULT_MODEL_DEVICE = "cuda:0"
DEFAULT_DECODER_DEVICE = "cuda:1"
LINEAR_CLA_TOKEN = "<linear_cla>"
LINEAR_PRE_TOKEN = "<linear_pre>"
PROT_CLA_TOKEN = "<prot_cla>"
PROTEIN_FOLD_TOKEN = "<prot_st>"
IMAGE_GEN_TOKEN = "<image_gen>"
IMAGE_EDIT_TOKEN = "<image_edit>"
SPECTRA_TOKEN = "<spectra_st>"
STATS_PATH = project_path("s1_omni_infer", "s1_omni_regression_label_stats.json")
ROUTE_TOKENS = (
    PROT_CLA_TOKEN,
    PROTEIN_FOLD_TOKEN,
    LINEAR_CLA_TOKEN,
    LINEAR_PRE_TOKEN,
    IMAGE_GEN_TOKEN,
    IMAGE_EDIT_TOKEN,
    SPECTRA_TOKEN,
)
IMAGE_CONFIG_NAME = "s1_omni_image_config.json"
DEFAULT_IMAGE_OUTPUT_DIR = project_path("pred_out", "images")
DEFAULT_SPECTRA_OUTPUT_DIR = project_path("pred_out", "spectra")
DEFAULT_PROTEIN_OUTPUT_DIR = project_path("pred_out", "protein")
DEFAULT_SPEC2MOL_CONFIG = qwen_path("spec2mol_v2", "config", "spec2mol_v2.yaml")
DEFAULT_PROTEIN_VIEWER_CDN = "https://cdn.jsdelivr.net/npm/3dmol/build/3Dmol-min.js"
DEFAULT_SIMPLEFOLD_OUTPUT_DIR = project_path("pred_out", "simplefold")
DEFAULT_SIMPLEFOLD_CCD_CACHE = project_path("simplefold_inference_pt")
DEFAULT_SIMPLEFOLD_TORCH_HUB = ""
SPECTRA_CONTEXT_INDEX = {"uv": 0, "ir": 1, "raman": 2}
SPECTRA_IMAGE_TARGET_SIZE = (1024, 552)
DEFAULT_SYSTEM_PROMPT = (
    "你是中国科学院磐石团队训练的 S1-Omni 统一架构多模态模型，"
    "具备多模态图像理解和生成能力。"
)
QWEN3_T2I_TEMPLATE = (
    "<|im_start|>system\n" + DEFAULT_SYSTEM_PROMPT
    + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWEN3_EDIT_TEMPLATE = (
    "<|im_start|>system\n" + DEFAULT_SYSTEM_PROMPT
    + "<|im_end|>\n<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stats_path", default=STATS_PATH)
    parser.add_argument("--question", default=None)
    parser.add_argument("--question_file", default=None)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--device", default=DEFAULT_MODEL_DEVICE)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--max_new_tokens", type=int, default=8196)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--protein_threshold", type=float, default=0.9)
    parser.add_argument("--pdb_files", nargs="+", default=None, help="PDB/mmCIF files for protein site prediction.")
    parser.add_argument("--protein_output_dir", default=DEFAULT_PROTEIN_OUTPUT_DIR)
    parser.add_argument("--protein_viewer_cdn", default=DEFAULT_PROTEIN_VIEWER_CDN)
    parser.add_argument("--protein_summary_max_new_tokens", type=int, default=2048)
    parser.add_argument("--strip_protein_for_generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_input", default=None, help="Input image path for <image_edit>.")
    parser.add_argument("--image_output_dir", default=DEFAULT_IMAGE_OUTPUT_DIR)
    parser.add_argument("--image_height", type=int, default=1024)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--image_steps", type=int, default=50)
    parser.add_argument("--image_edit_steps", type=int, default=40)
    parser.add_argument("--image_cfg", type=float, default=4.0)
    parser.add_argument("--image_seed", type=int, default=42)
    parser.add_argument("--image_device", default=DEFAULT_DECODER_DEVICE, help="Device for image DiT/alignment/VAE.")
    parser.add_argument("--image_dtype", default=None, choices=["bf16", "fp16", "fp32"], help="Image decoder dtype. Defaults to --dtype.")
    parser.add_argument("--image_config_name", default=IMAGE_CONFIG_NAME)
    parser.add_argument("--load_image_decoder_at_start", action="store_true")
    parser.add_argument("--simplefold_output_dir", default=DEFAULT_SIMPLEFOLD_OUTPUT_DIR)
    parser.add_argument("--simplefold_output_format", default="cif", choices=["cif", "pdb"])
    parser.add_argument("--simplefold_device", default=DEFAULT_DECODER_DEVICE, help="Device for SimpleFold.")
    parser.add_argument("--simplefold_dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--simplefold_ccd_cache_dir", default=DEFAULT_SIMPLEFOLD_CCD_CACHE)
    parser.add_argument("--simplefold_torch_hub_dir", default=DEFAULT_SIMPLEFOLD_TORCH_HUB)
    parser.add_argument("--simplefold_workspace", default=None)
    parser.add_argument("--simplefold_num_steps", type=int, default=500)
    parser.add_argument("--simplefold_tau", type=float, default=0.1)
    parser.add_argument("--simplefold_seed", type=int, default=42)
    parser.add_argument("--simplefold_sample_id", default=None)
    parser.add_argument("--simplefold_route_token_policy", default="last", choices=["first", "last"])
    parser.add_argument("--load_simplefold_decoder_at_start", action="store_true")
    parser.add_argument("--jdx_files", nargs="+", default=None, help="One or more IR/Raman/UV JDX files for <spectra_st>.")
    parser.add_argument("--spectra_output_dir", default=DEFAULT_SPECTRA_OUTPUT_DIR)
    parser.add_argument("--spec2mol_config", default=DEFAULT_SPEC2MOL_CONFIG)
    parser.add_argument("--spec2mol_device", default=DEFAULT_DECODER_DEVICE, help="Device for spec2mol decoder.")
    parser.add_argument("--ir_prompt", default=None)
    parser.add_argument("--raman_prompt", default=None)
    parser.add_argument("--uv_prompt", default=None)
    parser.add_argument("--spectra_max_image_side", type=int, default=None)
    parser.add_argument("--force_spectra", action="store_true", help="Run spec2mol decoder even if generate does not emit <spectra_st>.")
    parser.add_argument("--load_spec2mol_decoder_at_start", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def resolve_simplefold_dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


class S1OmniLinearPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        current_dim = hidden_dim
        for _ in range(num_layers - 2):
            next_dim = max(current_dim // 2, 1)
            self.layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim

        self.layers.append(nn.Linear(current_dim, 1))

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if index < len(self.layers) - 1 else layer(x)
        return x.flatten()


class AlignmentLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 5120,
        output_dim: int = 3584,
        hidden_dim: int | None = None,
        layer_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "mlp":
            if hidden_dim is None:
                raise ValueError("hidden_dim is required for image alignment MLP")
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim, bias=True),
            )
        elif layer_type == "linear":
            self.projection = nn.Linear(input_dim, output_dim, bias=True)
        elif layer_type == "mlp_ln":
            if hidden_dim is None:
                raise ValueError("hidden_dim is required for image alignment mlp_ln")
            self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
            self.ln2 = nn.LayerNorm(output_dim)
            self.act = nn.GELU()
        else:
            raise ValueError(f"unknown image alignment layer_type={layer_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "mlp":
            return self.net(x)
        if self.layer_type == "linear":
            return self.projection(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.ln2(x)


def apply_image_compat_patches() -> None:
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    try:
        if "enable_gqa" in inspect.signature(orig_sdpa).parameters:
            return
    except (TypeError, ValueError):
        pass

    if getattr(orig_sdpa, "_s1_omni_image_compat", False):
        return

    def sdpa_compat(query, key, value, *args, enable_gqa=False, **kwargs):
        if enable_gqa:
            n_q, n_kv = query.shape[-3], key.shape[-3]
            if n_kv and n_q % n_kv == 0 and n_q != n_kv:
                rep = n_q // n_kv
                key = key.repeat_interleave(rep, dim=-3)
                value = value.repeat_interleave(rep, dim=-3)
        return orig_sdpa(query, key, value, *args, **kwargs)

    sdpa_compat._s1_omni_image_compat = True
    torch.nn.functional.scaled_dot_product_attention = sdpa_compat


def pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )


def unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    batch_size, _num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height, width)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def resize_for_encoder(image, target_area: int = 1024 * 1024):
    ratio = image.size[0] / image.size[1]
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = max(round(width / 32) * 32, 32)
    height = max(round(height / 32) * 32, 32)
    return image.resize((int(width), int(height)))


def compute_drop_idx(tokenizer, template: str) -> int:
    prefix = template.split("{}", 1)[0]
    encoded = tokenizer(prefix, return_tensors="pt")
    return encoded["input_ids"].shape[1]


def _build_from_config(cls, cfg: dict[str, Any]):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        loaded = cls.load_config(tmpdir)
    with torch.device("meta"):
        return cls.from_config(loaded)


def materialize_meta_module(module: nn.Module, device: str, dtype: torch.dtype) -> nn.Module:
    for param in module.parameters():
        if not param.is_meta:
            param.data = param.data.to(
                device=device,
                dtype=dtype if param.is_floating_point() else param.dtype,
            )
    for buf in module.buffers():
        if not buf.is_meta:
            buf.data = buf.data.to(
                device=device,
                dtype=dtype if buf.is_floating_point() else buf.dtype,
            )
    for mod in module.modules():
        has_meta = any(t.is_meta for t in mod.parameters()) or any(
            t.is_meta for t in mod.buffers()
        )
        child_has_meta = any(
            any(t.is_meta for t in child.parameters())
            or any(t.is_meta for t in child.buffers())
            for child in mod.children()
        )
        if has_meta and not child_has_meta and hasattr(mod, "to_empty"):
            mod.to_empty(device=device)
    return module


def fix_qwen_image_rope_freqs(transformer: nn.Module) -> None:
    pos_embed = getattr(transformer, "pos_embed", None)
    if pos_embed is None:
        return
    freqs = getattr(pos_embed, "pos_freqs", None)
    if freqs is None or not getattr(freqs, "is_meta", False):
        return
    pos_index = torch.arange(4096)
    neg_index = torch.arange(4096).flip(0) * -1 - 1
    pos_embed.pos_freqs = torch.cat(
        [
            pos_embed.rope_params(pos_index, pos_embed.axes_dim[0], pos_embed.theta),
            pos_embed.rope_params(pos_index, pos_embed.axes_dim[1], pos_embed.theta),
            pos_embed.rope_params(pos_index, pos_embed.axes_dim[2], pos_embed.theta),
        ],
        dim=1,
    )
    pos_embed.neg_freqs = torch.cat(
        [
            pos_embed.rope_params(neg_index, pos_embed.axes_dim[0], pos_embed.theta),
            pos_embed.rope_params(neg_index, pos_embed.axes_dim[1], pos_embed.theta),
            pos_embed.rope_params(neg_index, pos_embed.axes_dim[2], pos_embed.theta),
        ],
        dim=1,
    )


def _modulate_with_index(
    x: torch.Tensor,
    mod_params: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift, scale, gate = mod_params.chunk(3, dim=-1)
    shift_0, shift_1 = shift[0:1], shift[1:2]
    scale_0, scale_1 = scale[0:1], scale[1:2]
    gate_0, gate_1 = gate[0:1], gate[1:2]
    idx = index.unsqueeze(-1)
    s = torch.where(idx == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
    sc = torch.where(idx == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
    g = torch.where(idx == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
    return x * (1 + sc) + s, g


def _modulate_simple(x: torch.Tensor, mod_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift, scale, gate = mod_params.chunk(3, dim=-1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)


class S1ImageDecoder:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        qwen_model: S1Protein,
        tokenizer,
        device: str,
        dtype: torch.dtype,
        config_name: str = IMAGE_CONFIG_NAME,
    ) -> None:
        apply_image_compat_patches()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.qwen_model = qwen_model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.vae_scale_factor = 8
        config_path = self.checkpoint_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"missing image config: {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.prefixes = self.config.get(
            "prefixes",
            {"alignment_mlp": "alignment_mlp.", "dit": "dit.", "vae": "vae."},
        )
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = self.index["weight_map"]
        self.gen_drop_idx = compute_drop_idx(tokenizer, QWEN3_T2I_TEMPLATE)
        self.edit_drop_idx = compute_drop_idx(tokenizer, QWEN3_EDIT_TEMPLATE) - 3
        self._shard_handles: dict[str, Any] = {}

        self.alignment_layer = self._load_alignment_layer()
        self.transformer = self._load_transformer()
        self.vae = self._load_vae()
        self.scheduler = self._build_scheduler()
        self.image_processor = self._build_image_processor()
        self.qwen_device = next(self.qwen_model.backbone.parameters()).device

    def close(self) -> None:
        self._shard_handles.clear()

    def _keys_for(self, prefix_name: str) -> list[str]:
        prefix = self.prefixes[prefix_name]
        keys = [key for key in self.weight_map if key.startswith(prefix)]
        if not keys:
            raise RuntimeError(f"no image keys found for prefix {prefix!r}")
        return keys

    def _get_tensor(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        if filename not in self._shard_handles:
            self._shard_handles[filename] = safe_open(
                str(self.checkpoint_dir / filename),
                framework="pt",
                device="cpu",
            )
        return self._shard_handles[filename].get_tensor(key)

    def _load_component(self, module: nn.Module, prefix_name: str, strict: bool = True) -> nn.Module:
        prefix = self.prefixes[prefix_name]
        state = {}
        for key in self._keys_for(prefix_name):
            state[key.removeprefix(prefix)] = self._get_tensor(key)
        result = module.load_state_dict(state, strict=strict, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"failed to load image {prefix_name}: "
                f"missing={result.missing_keys[:5]} unexpected={result.unexpected_keys[:5]}"
            )
        materialize_meta_module(module, self.device, self.dtype)
        module.eval()
        return module

    def _load_alignment_layer(self) -> nn.Module:
        cfg = self.config.get("alignment_layer") or self.config.get("alignment_mlp", {})
        with torch.device("meta"):
            module = AlignmentLayer(
                input_dim=int(cfg.get("input_dim", 5120)),
                output_dim=int(cfg.get("output_dim", 3584)),
                hidden_dim=cfg.get("hidden_dim"),
                layer_type=cfg.get("type", "mlp"),
            )
        return self._load_component(module, "alignment_mlp")

    def _load_transformer(self) -> nn.Module:
        from diffusers import QwenImageTransformer2DModel

        module = _build_from_config(
            QwenImageTransformer2DModel,
            self.config["transformer_config"],
        )
        self._load_component(module, "dit")
        fix_qwen_image_rope_freqs(module)
        return module

    def _load_vae(self) -> nn.Module:
        from diffusers import AutoencoderKLQwenImage

        module = _build_from_config(AutoencoderKLQwenImage, self.config["vae_config"])
        return self._load_component(module, "vae")

    def _build_scheduler(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        return FlowMatchEulerDiscreteScheduler.from_config(self.config["scheduler_config"])

    def _build_image_processor(self):
        from diffusers.image_processor import VaeImageProcessor

        return VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    @torch.no_grad()
    def _forward_qwen_with_alignment(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.qwen_model.backbone.model(
            input_ids=input_ids.to(self.qwen_device),
            attention_mask=attention_mask.to(self.qwen_device),
            output_hidden_states=True,
        )
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        else:
            last_hidden = outputs[0]
        align_param = next(self.alignment_layer.parameters())
        return self.alignment_layer(
            last_hidden.to(device=align_param.device, dtype=align_param.dtype)
        )

    @torch.no_grad()
    def encode_output_ids(self, generated_ids: torch.Tensor, drop_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        full_mask = torch.ones_like(generated_ids)
        aligned = self._forward_qwen_with_alignment(generated_ids, full_mask)
        valid_len = int(full_mask[0].sum().item())
        if drop_idx >= valid_len:
            drop_idx = 0
        feat = aligned[0, drop_idx:valid_len, :].unsqueeze(0)
        if feat.size(1) <= 0:
            raise ValueError(f"empty image conditioning: drop_idx={drop_idx}, valid_len={valid_len}")
        dit_param = next(self.transformer.parameters())
        feat = feat.to(device=dit_param.device, dtype=dit_param.dtype)
        mask = torch.ones(1, feat.size(1), dtype=torch.long, device=dit_param.device)
        return feat, mask

    @torch.inference_mode()
    def encode_negative_prompt(self) -> tuple[torch.Tensor, torch.Tensor]:
        neg_text = QWEN3_T2I_TEMPLATE.format("")
        inputs = self.tokenizer(
            neg_text,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.qwen_device)
        attention_mask = inputs["attention_mask"].to(self.qwen_device)
        aligned = self._forward_qwen_with_alignment(input_ids, attention_mask)
        valid_len = int(attention_mask[0].sum().item())
        drop_idx = self.gen_drop_idx if self.gen_drop_idx < valid_len else 0
        features = aligned[:, drop_idx:valid_len, :]
        mask = torch.ones(1, features.size(1), dtype=torch.long, device=features.device)
        dit_param = next(self.transformer.parameters())
        return features.to(device=dit_param.device, dtype=dit_param.dtype), mask.to(dit_param.device)

    @torch.inference_mode()
    def text_to_image(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        height: int,
        width: int,
        steps: int,
        cfg: float,
        seed: int,
    ):
        from diffusers.utils.torch_utils import randn_tensor

        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        height = (height // 16) * 16
        width = (width // 16) * 16
        do_cfg = cfg > 1.0
        if do_cfg:
            neg_embeds, neg_mask = self.encode_negative_prompt()
            neg_txt_seq_lens = neg_mask.sum(dim=1).tolist()

        latent_height = 2 * (height // (self.vae_scale_factor * 2))
        latent_width = 2 * (width // (self.vae_scale_factor * 2))
        num_channels_latents = 16
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = randn_tensor(
            (1, 1, num_channels_latents, latent_height, latent_width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents = pack_latents(latents, 1, num_channels_latents, latent_height, latent_width)
        self.scheduler.set_timesteps(steps, device=device, mu=calculate_shift(latents.shape[1]))
        img_shapes = [[(1, latent_height // 2, latent_width // 2)]]
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        for t in self.scheduler.timesteps:
            timestep = t.expand(latents.shape[0]) / 1000.0
            noise_pred_pos = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                txt_seq_lens=txt_seq_lens,
                img_shapes=img_shapes,
                return_dict=False,
            )[0]
            if do_cfg:
                noise_pred_neg = self.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states=neg_embeds,
                    encoder_hidden_states_mask=neg_mask,
                    txt_seq_lens=neg_txt_seq_lens,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_neg + cfg * (noise_pred_pos - noise_pred_neg)
            else:
                noise_pred = noise_pred_pos
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return self.decode_latents(unpack_latents(latents, height, width, self.vae_scale_factor))

    @torch.inference_mode()
    def dit_forward_zero_cond_t(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list,
        txt_seq_lens: list,
        output_seq_len: int,
    ) -> torch.Tensor:
        dit = self.transformer
        hidden_states = dit.img_in(hidden_states)
        encoder_hidden_states = dit.txt_norm(encoder_hidden_states)
        encoder_hidden_states = dit.txt_in(encoder_hidden_states)
        timestep_doubled = torch.cat([timestep, timestep * 0], dim=0).to(hidden_states.dtype)
        temb_doubled = dit.time_text_embed(timestep_doubled, hidden_states)
        temb_real = temb_doubled[0:1]
        n_output = math.prod(img_shapes[0][0])
        n_cond = sum(math.prod(shape) for shape in img_shapes[0][1:])
        modulate_index = torch.tensor(
            [[0] * n_output + [1] * n_cond],
            device=hidden_states.device,
            dtype=torch.int,
        )
        image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        for block in dit.transformer_blocks:
            img_mod1, img_mod2 = block.img_mod(temb_doubled).chunk(2, dim=-1)
            txt_mod1, txt_mod2 = block.txt_mod(temb_real).chunk(2, dim=-1)
            img_modulated, img_gate = _modulate_with_index(
                block.img_norm1(hidden_states),
                img_mod1,
                modulate_index,
            )
            txt_modulated, txt_gate = _modulate_simple(
                block.txt_norm1(encoder_hidden_states),
                txt_mod1,
            )
            img_attn_out, txt_attn_out = block.attn(
                hidden_states=img_modulated,
                encoder_hidden_states=txt_modulated,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + img_gate * img_attn_out
            encoder_hidden_states = encoder_hidden_states + txt_gate * txt_attn_out
            img_modulated2, img_gate2 = _modulate_with_index(
                block.img_norm2(hidden_states),
                img_mod2,
                modulate_index,
            )
            hidden_states = hidden_states + img_gate2 * block.img_mlp(img_modulated2)
            txt_modulated2, txt_gate2 = _modulate_simple(
                block.txt_norm2(encoder_hidden_states),
                txt_mod2,
            )
            encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.txt_mlp(txt_modulated2)
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clamp(min=-65504, max=65504)
        hidden_states = dit.norm_out(hidden_states, temb_real)
        return dit.proj_out(hidden_states)[:, :output_seq_len]

    @torch.inference_mode()
    def image_editing(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        image,
        height: int,
        width: int,
        steps: int,
        seed: int,
    ):
        from diffusers.utils.torch_utils import randn_tensor

        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        vae_device = next(self.vae.parameters()).device
        input_image = image.resize((width, height))
        image_tensor = self.image_processor.preprocess(input_image).to(device=vae_device, dtype=dtype)
        image_tensor = image_tensor.unsqueeze(2)
        image_latents = self.vae.encode(image_tensor).latent_dist.sample()
        image_latents = self.normalise_latents(image_latents, image_latents.dtype)[:, :, 0].to(device)

        latent_height = 2 * (height // (self.vae_scale_factor * 2))
        latent_width = 2 * (width // (self.vae_scale_factor * 2))
        num_channels_latents = 16
        image_latents_packed = pack_latents(
            image_latents,
            1,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        noise_latents = randn_tensor(
            (1, 1, num_channels_latents, latent_height, latent_width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents = pack_latents(noise_latents, 1, num_channels_latents, latent_height, latent_width)
        output_seq_len = latents.shape[1]
        self.scheduler.set_timesteps(steps, device=device, mu=calculate_shift(output_seq_len))
        img_h = latent_height // 2
        img_w = latent_width // 2
        img_shapes = [[(1, img_h, img_w), (1, img_h, img_w)]]
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        for t in self.scheduler.timesteps:
            timestep = t.expand(latents.shape[0]) / 1000.0
            latent_model_input = torch.cat([latents, image_latents_packed], dim=1)
            noise_pred = self.dit_forward_zero_cond_t(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                output_seq_len=output_seq_len,
            )
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return self.decode_latents(unpack_latents(latents, height, width, self.vae_scale_factor))

    def normalise_latents(self, latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, dtype)
        )
        return (latents - latents_mean) / latents_std

    def denormalise_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        inv_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        return latents / inv_std + latents_mean

    @torch.inference_mode()
    def decode_latents(self, latents: torch.Tensor):
        vae_device = next(self.vae.parameters()).device
        if latents.device != vae_device:
            latents = latents.to(vae_device)
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)
        image = self.vae.decode(self.denormalise_latents(latents), return_dict=False)[0]
        if image.dim() == 5:
            image = image[:, :, 0]
        return self.image_processor.postprocess(image, output_type="pil")[0]

    def infer_from_generated(
        self,
        route_token: str,
        generated_ids: torch.Tensor,
        condition_drop_idx: int,
        source_image,
        args,
    ):
        is_edit = route_token == IMAGE_EDIT_TOKEN and source_image is not None
        drop_idx = condition_drop_idx if condition_drop_idx > 0 else (self.edit_drop_idx if is_edit else self.gen_drop_idx)
        prompt_embeds, prompt_embeds_mask = self.encode_output_ids(generated_ids, drop_idx)
        if is_edit:
            image = self.image_editing(
                prompt_embeds,
                prompt_embeds_mask,
                source_image,
                height=args.image_height,
                width=args.image_width,
                steps=args.image_edit_steps,
                seed=args.image_seed,
            )
            task = "image_edit"
        else:
            image = self.text_to_image(
                prompt_embeds,
                prompt_embeds_mask,
                height=args.image_height,
                width=args.image_width,
                steps=args.image_steps,
                cfg=args.image_cfg,
                seed=args.image_seed,
            )
            task = "image_gen"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return image, task


def build_messages(question: str):
    return [{"role": "user", "content": [{"type": "text", "text": question}]}]


def read_question(args) -> str:
    if args.question_file:
        return Path(args.question_file).read_text(encoding="utf-8").strip()
    if args.question is None:
        raise ValueError("missing question")
    return args.question


def _text_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    for key in ("input", "text", "question"):
        if key in item:
            return str(item[key])
    if "messages" in item:
        parts = []
        for msg in item["messages"]:
            if msg.get("role") == "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type", "text") == "text"
                )
        text = "".join(parts).strip()
        if text:
            return text
    raise KeyError("batch item must be a string or contain input/text/question/messages")


def read_questions(path: str | Path) -> list[str]:
    input_path = Path(path)
    if input_path.suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        return [_text_from_item(item) for item in data]

    questions = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(_text_from_item(json.loads(line)))
            except json.JSONDecodeError:
                questions.append(line)
    return questions


def prepare_generation_question(question: str, use_esm2: bool, strip_protein: bool) -> str:
    if not strip_protein or not use_esm2 or "<PROT>" not in question:
        return question
    try:
        qwen_text, _ = _extract_protein_sequence_and_qwen_text(question)
        return qwen_text
    except Exception:
        return question


def prepare_generation_inputs(tokenizer, question: str, device: str):
    inputs = tokenizer.apply_chat_template(
        build_messages(question),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def prepare_image_generation_inputs(tokenizer, processor, question: str, image_path: str, device: str):
    image = load_pil_image(image_path)
    if processor is None:
        raise RuntimeError("Qwen3VLProcessor is required for image editing generation inputs")
    text = [QWEN3_EDIT_TEMPLATE.format(question)]
    inputs = processor(
        text=text,
        images=[resize_for_encoder(image)],
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _stage_jdx_files(jdx_files: list[str], temp_root: Path) -> Path:
    input_dir = temp_root / "jdx"
    input_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for index, raw in enumerate(jdx_files, start=1):
        src = Path(raw).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"JDX file does not exist: {src}")
        name = src.name
        if name in seen:
            name = f"{index:02d}_{name}"
        seen.add(name)
        dst = input_dir / name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return input_dir


def verify_zero_filled_spectra_context(bundle) -> None:
    for modality in bundle.zero_filled_spectra:
        idx = SPECTRA_CONTEXT_INDEX[modality]
        tensor = bundle.context[idx]
        if torch.count_nonzero(tensor).item() != 0:
            raise RuntimeError(f"missing {modality} spectra context is not zero-filled")


def prepare_spectra_context(args, temp_root: Path) -> dict[str, Any]:
    if not args.jdx_files:
        raise ValueError("--jdx_files is required for spectra inference")
    from spec2mol_v2.jdx import load_spectra_dir
    try:
        from spec2mol_v2.spectra_plot import render_spectra_images
    except ImportError as exc:
        raise RuntimeError(
            "spec2mol spectra rendering requires matplotlib; install/use the "
            "spec2mol_v2 inference environment before running --jdx_files"
        ) from exc

    input_dir = _stage_jdx_files(args.jdx_files, temp_root)
    bundle = load_spectra_dir(input_dir)
    verify_zero_filled_spectra_context(bundle)
    image_meta = render_spectra_images(
        input_dir,
        temp_root / "spectra_images",
        max_image_side=args.spectra_max_image_side,
    )
    all_images = {
        modality: Path(path)
        for modality, path in image_meta.get("image_files", {}).items()
    }
    present_images = {
        modality: all_images[modality]
        for modality in bundle.present_spectra
        if modality in all_images
    }
    if not present_images:
        raise RuntimeError(f"no spectra images were rendered from {args.jdx_files}")
    return {
        "input_dir": input_dir,
        "bundle": bundle,
        "image_meta": image_meta,
        "present_images": present_images,
        "jdx_files": [str(Path(path).expanduser().resolve()) for path in args.jdx_files],
    }


def _load_rgb_image(path: str | Path, target_size: tuple[int, int] | None = None):
    from PIL import Image

    with Image.open(path) as image:
        loaded = image.convert("RGB").copy()
    if target_size is not None and loaded.size != target_size:
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        loaded = loaded.resize(target_size, resample=resample)
    return loaded


def _build_spectra_messages(images: list[Any], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _apply_processor_template(processor, messages: list[dict[str, Any]]) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def prepare_spectra_generation_inputs(processor, question: str, spectra_context: dict[str, Any], device: str):
    if processor is None:
        raise RuntimeError("Qwen3VLProcessor is required for spectra generation inputs")
    present_images = spectra_context["present_images"]
    image_order = [modality for modality in ("ir", "raman", "uv") if modality in present_images]
    images = [_load_rgb_image(present_images[modality]) for modality in image_order]
    text = _apply_processor_template(processor, _build_spectra_messages(images, question))
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=False,
    )
    spectra_context["image_order"] = image_order
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def load_qwen3_vl_processor(checkpoint_dir: str | Path):
    try:
        from transformers import Qwen3VLProcessor
    except ImportError as exc:
        raise RuntimeError("transformers.Qwen3VLProcessor is required for --image_input") from exc

    processor = Qwen3VLProcessor.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = 256 * 28 * 28
        processor.image_processor.max_pixels = 1280 * 28 * 28
    return processor


def prepare_protein_inputs(tokenizer, question: str, device: str, esm_tokenizer=None, use_esm2: bool = True):
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_esm2:
        normalized_question, protein_sequence = _extract_protein_sequence_and_qwen_text(question)
    else:
        normalized_question, protein_sequence = _space_protein_sequence(question)

    text = tokenizer.apply_chat_template(
        build_messages(normalized_question),
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    if use_esm2:
        if esm_tokenizer is None:
            raise ValueError("esm_tokenizer is required when use_esm2=True")
        esm_inputs = esm_tokenizer([protein_sequence], padding=True, return_tensors="pt")
        inputs["esm_input_ids"] = esm_inputs["input_ids"]
        inputs["esm_attention_mask"] = esm_inputs["attention_mask"]
    else:
        protein_token_mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        residue_token_ids = tokenizer.encode(" " + " ".join(protein_sequence), add_special_tokens=False)
        residue_start = find_subsequence(inputs["input_ids"][0].tolist(), residue_token_ids)
        if residue_start < 0:
            raise ValueError("failed to locate spaced protein residue tokens in tokenized input")
        protein_token_mask[0, residue_start : residue_start + len(residue_token_ids)] = True
        inputs["protein_token_mask"] = protein_token_mask

    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    return model_inputs, len(protein_sequence)


def extract_simplefold_sequence(question: str) -> str:
    try:
        _, sequence = _extract_protein_sequence_and_qwen_text(question)
        return sequence
    except Exception:
        pass

    compact = re.sub(r"\s+", "", question.upper())
    candidates = re.findall(r"[ACDEFGHIKLMNPQRSTVWY:]{10,}", compact)
    if not candidates:
        raise ValueError(
            "could not infer an amino-acid sequence for SimpleFold; "
            "include it in <PROT>...</PROT> or pass a prompt containing a plain AA sequence"
        )
    return max(candidates, key=len)


def load_regression_stats(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"normalization": "none", "enabled": False, "mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "range": 1.0}
    stats_path = Path(path)
    if not stats_path.exists():
        return {"normalization": "none", "enabled": False, "mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "range": 1.0}
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    normalization = str(stats.get("normalization", "zscore")).lower()
    enabled = bool(stats.get("enabled", normalization != "none"))
    label_min = float(stats.get("min", 0.0))
    label_max = float(stats.get("max", label_min + float(stats.get("range", 1.0))))
    label_range = float(stats.get("range", label_max - label_min))
    if abs(label_range) < 1e-6:
        label_range = 1.0
    std = float(stats.get("std", 1.0))
    if abs(std) < 1e-6:
        std = 1.0
    return {
        "normalization": normalization,
        "enabled": enabled,
        "mean": float(stats.get("mean", 0.0)),
        "std": std,
        "min": label_min,
        "max": label_max,
        "range": label_range,
    }


def denormalize_regression(value: float, stats: dict[str, Any]) -> float:
    if not stats.get("enabled", True):
        return value
    normalization = str(stats.get("normalization", "zscore")).lower()
    if normalization == "minmax":
        return value * stats["range"] + stats["min"]
    if normalization in {"none", "identity"}:
        return value
    return value * stats["std"] + stats["mean"]


def load_predictor_state(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing model.safetensors.index.json in {checkpoint_dir}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    predictor_keys = sorted(key for key in weight_map if key.startswith("predictor."))
    if not predictor_keys:
        raise RuntimeError(f"no predictor.* keys found in {index_path}")

    state: dict[str, torch.Tensor] = {}
    for filename in sorted({weight_map[key] for key in predictor_keys}):
        shard = safe_load_file(str(checkpoint_dir / filename), device="cpu")
        for key in predictor_keys:
            if weight_map[key] == filename and key in shard:
                state[key.removeprefix("predictor.")] = shard[key]
    if not state:
        raise RuntimeError(f"failed to load predictor tensors from {checkpoint_dir}")
    return state


def build_property_predictor(checkpoint_dir: str | Path, hidden_size: int, device: str, dtype: torch.dtype):
    state = load_predictor_state(checkpoint_dir)
    layer_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("layers.") and key.split(".")[1].isdigit()
    }
    num_layers = max(layer_indices) + 1 if layer_indices else 4
    predictor = S1OmniLinearPredictor(
        input_dim=hidden_size,
        hidden_dim=hidden_size,
        num_layers=num_layers,
    )
    predictor.load_state_dict(state, strict=True)
    predictor.to(device=device, dtype=dtype)
    predictor.eval()
    return predictor


def _token_ids_for_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def find_subsequence(row: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    max_start = len(row) - len(pattern)
    for index in range(start, max_start + 1):
        if row[index : index + len(pattern)] == pattern:
            return index
    return -1


def _find_chat_role_token_spans(tokenizer, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], role: str):
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    role_ids = _token_ids_for_text(tokenizer, role)
    newline_ids = _token_ids_for_text(tokenizer, "\n")
    if im_start_id is None or im_end_id is None or not role_ids:
        return None

    spans = []
    lengths = (
        attention_mask.long().sum(dim=-1).clamp(min=1).tolist()
        if attention_mask is not None
        else [input_ids.size(1)] * input_ids.size(0)
    )
    header_pattern = [im_start_id] + role_ids
    for row, length in zip(input_ids, lengths):
        row_list = row[: int(length)].tolist()
        search_from = 0
        selected_span = None
        while True:
            header_pos = find_subsequence(row_list, header_pattern, search_from)
            if header_pos < 0:
                break
            content_start = header_pos + len(header_pattern)
            if newline_ids and row_list[content_start : content_start + len(newline_ids)] == newline_ids:
                content_start += len(newline_ids)
            end_pos = find_subsequence(row_list, [im_end_id], content_start)
            if end_pos < 0:
                break
            if end_pos > content_start:
                selected_span = (content_start, end_pos)
            search_from = end_pos + 1
        spans.append(selected_span)
    return spans


def pool_question_hidden(tokenizer, hidden_states: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    spans = _find_chat_role_token_spans(tokenizer, input_ids, attention_mask, role="user")
    if spans is None:
        lengths = attention_mask.long().sum(dim=-1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    pooled_states = []
    fallback_lengths = attention_mask.long().sum(dim=-1).clamp(min=1) - 1
    for index, span in enumerate(spans):
        if span is None:
            pooled_states.append(hidden_states[index, int(fallback_lengths[index].item())])
        else:
            start, end = span
            pooled_states.append(hidden_states[index, int(start) : int(end)].mean(dim=0))
    return torch.stack(pooled_states, dim=0)


def route_token_id(tokenizer, route_token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(route_token)
    if token_id is None or token_id < 0 or token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"route token is not registered as a tokenizer token: {route_token!r}")
    return int(token_id)


def select_route_token_hidden(
    tokenizer,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    route_token: str,
    policy: str,
) -> torch.Tensor:
    input_ids_on_device = input_ids.to(device=hidden_states.device)
    positions = None
    try:
        token_id = route_token_id(tokenizer, route_token)
        mask = input_ids_on_device.eq(token_id)
        positions = mask[0].nonzero(as_tuple=False).flatten()
    except RuntimeError:
        positions = None

    if positions is not None and int(positions.numel()) >= 1:
        pos = positions[0] if policy == "first" else positions[-1]
    else:
        pattern = tokenizer.encode(route_token, add_special_tokens=False)
        start = find_subsequence(input_ids[0].tolist(), pattern)
        if start < 0:
            raise RuntimeError(f"route token {route_token!r} was not found in prefill input")
        pos = torch.tensor(
            start if policy == "first" else start + len(pattern) - 1,
            device=hidden_states.device,
        )
    return hidden_states[0, int(pos.item())].detach().float().cpu()


def extract_final_route_token(tokenizer, generated_new_ids: torch.Tensor, generated_text: str) -> str | None:
    token_id_to_route = {}
    for token in ROUTE_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            token_id_to_route[int(token_id)] = token
    for token_id in reversed(generated_new_ids.tolist()):
        route = token_id_to_route.get(int(token_id))
        if route is not None:
            return route

    route_pattern = "|".join(re.escape(token.strip("<>")) for token in ROUTE_TOKENS)
    matches = list(re.finditer(rf"<(?:{route_pattern})>", generated_text))
    return matches[-1].group(0) if matches else None


def replace_last_route_token(text: str, route_token: str | None, replacement: str) -> str:
    if route_token is None:
        return text
    index = text.rfind(route_token)
    if index < 0:
        return text
    return text[:index] + replacement + text[index + len(route_token):]


class LazyImageDecoder:
    def __init__(self, checkpoint_dir: str | Path, model: S1Protein, tokenizer, args) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.decoder: S1ImageDecoder | None = None

    def get(self) -> S1ImageDecoder:
        if self.decoder is None:
            image_device = self.args.image_device or DEFAULT_DECODER_DEVICE
            image_dtype = resolve_dtype(self.args.image_dtype or self.args.dtype)
            self.decoder = S1ImageDecoder(
                self.checkpoint_dir,
                self.model,
                self.tokenizer,
                device=image_device,
                dtype=image_dtype,
                config_name=self.args.image_config_name,
            )
        return self.decoder


class LazySimpleFoldDecoder:
    def __init__(self, checkpoint_dir: str | Path, args) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.args = args
        self.decoder: nn.Module | None = None

    def get(self) -> nn.Module:
        if self.decoder is None:
            try:
                from s1_omni.modeling_simplefold import SimpleFold
            except ImportError as exc:
                raise RuntimeError(
                    "SimpleFold route was selected, but qwenvl.modeling_simplefold "
                    "is not available in this workspace. Bring in "
                    "qwen-vl-finetune/qwenvl/modeling_simplefold.py and the "
                    "qwenvl/protein_folding package from feature/simplefold-inference."
                ) from exc

            torch_hub_dir = self.args.simplefold_torch_hub_dir
            if os.environ.get("SIMPLEFOLD_TORCH_HUB_DIR") is None and torch_hub_dir:
                os.environ["SIMPLEFOLD_TORCH_HUB_DIR"] = str(torch_hub_dir)

            device = self.args.simplefold_device or DEFAULT_DECODER_DEVICE
            self.decoder = SimpleFold.from_pretrained(
                self.checkpoint_dir,
                dtype=resolve_simplefold_dtype(self.args.simplefold_dtype),
                device=device,
            )
            if self.args.simplefold_ccd_cache_dir:
                self.decoder.set_ccd_cache_dir(self.args.simplefold_ccd_cache_dir)
            if not getattr(self.decoder, "is_text_conditioned", False):
                raise RuntimeError("the selected SimpleFold checkpoint is not text-conditioned")
        return self.decoder


class LazySpec2MolDecoder:
    def __init__(self, checkpoint_dir: str | Path, args) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.args = args
        self.inline: dict[str, Any] | None = None
        self.bank = None
        self.model = None

    def get_modules(self) -> dict[str, Any]:
        if self.inline is None:
            from spec2mol_v2.hf_modules_loader import load_hf_inline_modules

            self.inline = load_hf_inline_modules(self.checkpoint_dir, device="cpu")
        return self.inline["modules"]

    def get_bank(self):
        if self.bank is None:
            from spec2mol_v2.s1_omni_linear_predictor import S1OmniLinearPredictorBank

            self.bank = S1OmniLinearPredictorBank.from_state_dicts(
                self.get_modules(),
                device=self.args.spec2mol_device,
                dtype=resolve_dtype(self.args.dtype),
            )
        return self.bank

    def get_model(self):
        if self.model is None:
            from spec2mol_v2.model import Spec2MolModel

            self.model = Spec2MolModel.from_inline_modules(
                self.args.spec2mol_config,
                self.get_modules(),
                device=self.args.spec2mol_device,
            )
        return self.model


def load_pil_image(path: str | Path | None):
    if path is None:
        return None
    from PIL import Image

    return Image.open(path).convert("RGB")


def save_generated_image(image, output_dir: str | Path, index: int | None, task: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_{index:06d}" if index is not None else task
    path = output_path / f"{stem}.png"
    suffix = 1
    while path.exists():
        path = output_path / f"{stem}_{suffix}.png"
        suffix += 1
    image.save(path)
    return str(path)


def safe_filename_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    stem = stem.strip("._")
    return stem or "sample"


def save_simplefold_structure(
    simplefold_model,
    output,
    output_dir: str | Path,
    sample_id: str,
    item_index: int | None,
    output_format: str,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = ".pdb" if output_format == "pdb" else ".cif"
    stem = safe_filename_stem(sample_id)
    if item_index is not None:
        stem = f"{stem}_{item_index:06d}"
    path = output_path / f"{stem}{suffix}"
    counter = 1
    while path.exists():
        path = output_path / f"{stem}_{counter}{suffix}"
        counter += 1

    if output_format == "pdb":
        saved = simplefold_model.save_pdb(output, path)
    else:
        saved = simplefold_model.save_cif(output, path)
    return str(saved)


def load_dual_model(args):
    dtype = resolve_dtype(args.dtype)
    model = S1Protein.from_pretrained(
        args.checkpoint_dir,
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir,
        padding_side="right",
        use_fast=False,
    )
    processor = load_qwen3_vl_processor(args.checkpoint_dir) if (args.image_input or args.jdx_files) else None
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.set_tokenizer(tokenizer)
    model.to(args.device)
    model.eval()

    esm_tokenizer = None
    if getattr(model, "use_esm2", False):
        esm_source = S1Protein.resolve_esm_tokenizer_source(args.checkpoint_dir, model.esm_model_name)
        esm_tokenizer = AutoTokenizer.from_pretrained(esm_source)

    predictor = build_property_predictor(
        args.checkpoint_dir,
        hidden_size=model.hidden_size,
        device=args.device,
        dtype=next(model.backbone.parameters()).dtype,
    )
    image_decoder = LazyImageDecoder(args.checkpoint_dir, model, tokenizer, args)
    if args.load_image_decoder_at_start:
        image_decoder.get()
    simplefold_decoder = LazySimpleFoldDecoder(args.checkpoint_dir, args)
    if args.load_simplefold_decoder_at_start:
        simplefold_decoder.get()
    spec2mol_decoder = LazySpec2MolDecoder(args.checkpoint_dir, args)
    if args.load_spec2mol_decoder_at_start:
        spec2mol_decoder.get_bank()
        spec2mol_decoder.get_model()
    return model, tokenizer, processor, esm_tokenizer, predictor, image_decoder, simplefold_decoder, spec2mol_decoder


def generate_route(
    model: S1Protein,
    tokenizer,
    processor,
    question: str,
    args,
    spectra_context: dict[str, Any] | None = None,
):
    # generation_question = prepare_generation_question(
    #     question,
    #     use_esm2=getattr(model, "use_esm2", False),
    #     strip_protein=args.strip_protein_for_generation,
    # )
    generation_question = question
    if spectra_context is not None:
        inputs = prepare_spectra_generation_inputs(
            processor,
            generation_question,
            spectra_context,
            args.device,
        )
    elif args.image_input:
        inputs = prepare_image_generation_inputs(
            tokenizer,
            processor,
            generation_question,
            args.image_input,
            args.device,
        )
    else:
        inputs = prepare_generation_inputs(tokenizer, generation_question, args.device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p if args.temperature > 0 else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    generated_ids = model.backbone.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    user_spans = _find_chat_role_token_spans(
        tokenizer,
        inputs["input_ids"],
        inputs.get("attention_mask"),
        role="user",
    )
    condition_drop_idx = (
        int(user_spans[0][0])
        if user_spans and user_spans[0] is not None
        else prompt_len
    )
    new_ids = generated_ids[0, prompt_len:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=False).strip()
    route_token = extract_final_route_token(tokenizer, new_ids, generated_text)
    return generation_question, inputs, generated_ids, generated_text, route_token, condition_drop_idx


def infer_protein(model: S1Protein, tokenizer, esm_tokenizer, question: str, args):
    inputs, protein_length = prepare_protein_inputs(
        tokenizer,
        question,
        args.device,
        esm_tokenizer=esm_tokenizer,
        use_esm2=getattr(model, "use_esm2", False),
    )
    with torch.inference_mode():
        outputs = model(**inputs, threshold=args.protein_threshold)
    probabilities = outputs.probabilities[0].squeeze(-1)[:protein_length]
    probs = [float(x) for x in probabilities.detach().float().cpu().tolist()]
    bits = ["1" if value >= args.protein_threshold else "0" for value in probs]
    positive_indices = [index + 1 for index, bit in enumerate(bits) if bit == "1"]
    return {
        "bit_string": "".join(bits),
        "positive_indices": positive_indices,
        "probabilities": probs,
        "threshold": args.protein_threshold,
    }


def build_pdb_chain_records(pdb_files: list[str], question: str) -> list[dict[str, Any]]:
    from tools.pdb_to_sequence import extract_sequences

    records: list[dict[str, Any]] = []
    for raw_path in pdb_files:
        pdb_path = Path(raw_path).expanduser().resolve()
        if not pdb_path.is_file():
            raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")
        chains = extract_sequences(pdb_path)
        if not chains:
            raise ValueError(f"no standard amino acid chains found in {pdb_path}")
        pdb_id = pdb_path.stem
        for chain_id in sorted(chains.keys()):
            residues = chains[chain_id]
            sequence = "".join(residue["one"] for residue in residues)
            entry_id = f"{pdb_id}_{chain_id}"
            records.append(
                {
                    "id": entry_id,
                    "pdb_id": pdb_id,
                    "pdb_file": str(pdb_path),
                    "chain": chain_id,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "residues": residues,
                    "input": f"{question}\nSequence: <PROT>{sequence}</PROT>",
                }
            )
    return records


def _format_site_positions(positions: list[int]) -> str:
    return "[" + ", ".join(str(pos) for pos in positions) + "]"


def format_protein_chain_answer(chain_result: dict[str, Any]) -> str:
    record = chain_result["record"]
    positions = chain_result["prediction"]["positive_indices"]
    return (
        f"Protein {record['pdb_id']} chain {record['chain']} predicted binding-site "
        f"positions. The actual residue index of the site is: {_format_site_positions(positions)}"
    )


def save_protein_prediction_jsonl(
    chain_results: list[dict[str, Any]],
    output_dir: str | Path,
    item_index: int | None,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = "protein_sites" if item_index is None else f"protein_sites_{item_index:06d}"
    path = output_path / f"{stem}_{uuid.uuid4().hex[:8]}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for result in chain_results:
            record = result["record"]
            payload = {
                "id": record["id"],
                "messages": [
                    {"role": "user", "content": record["input"]},
                    {"role": "assistant", "content": result["answer_text"]},
                ],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(path.resolve())


def build_protein_viewers(pdb_files: list[str], prediction_jsonl: str, cdn_url: str) -> list[dict[str, Any]]:
    try:
        from tools.viewer import render_structure_3dmol
    except ImportError as exc:
        raise RuntimeError("viewer.py and its dependencies are required for protein visualization") from exc

    viewers: list[dict[str, Any]] = []
    for raw_path in pdb_files:
        pdb_path = Path(raw_path).expanduser().resolve()
        viewers.append(
            {
                "pdb_id": pdb_path.stem,
                "pdb_file": str(pdb_path),
                "html": render_structure_3dmol(str(pdb_path), prediction_jsonl, cdn_url),
            }
        )
    return viewers


def summarize_protein_chain_answers(
    model: S1Protein,
    tokenizer,
    chain_results: list[dict[str, Any]],
    args,
) -> str:
    if len(chain_results) == 1:
        return chain_results[0]["answer_text"]

    try:
        from tools.build_summary_prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT_TEMPLATE
    except ImportError as exc:
        raise RuntimeError("build_summary_prompts.py is required for multi-chain protein summaries") from exc

    parts: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in chain_results:
        grouped.setdefault(result["record"]["pdb_id"], []).append(result)
    for pdb_id in sorted(grouped):
        parts.append(f"### Protein: {pdb_id}")
        for result in sorted(grouped[pdb_id], key=lambda item: item["record"]["chain"]):
            record = result["record"]
            parts.append(f"\n**Chain {record['chain']}** ({record['id']}):\n")
            parts.append(result["answer_text"])
            parts.append("\n---\n")

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SUMMARY_USER_PROMPT_TEMPLATE.format(chain_predictions="\n".join(parts)),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.protein_summary_max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p if args.temperature > 0 else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    generated_ids = model.backbone.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True).strip()


def infer_protein_from_pdb(
    model: S1Protein,
    tokenizer,
    processor,
    esm_tokenizer,
    question: str,
    args,
    item_index: int | None = None,
) -> dict[str, Any]:
    chain_records = build_pdb_chain_records(args.pdb_files, question)
    chain_results: list[dict[str, Any]] = []
    for record in chain_records:
        with torch.inference_mode():
            generation_question, _, generated_ids, generated_text, route_token, _ = generate_route(
                model,
                tokenizer,
                processor,
                record["input"],
                args,
            )
        route_fallback = route_token is None
        effective_route_token = PROT_CLA_TOKEN if route_fallback else route_token
        if effective_route_token != PROT_CLA_TOKEN:
            raise RuntimeError(
                f"PDB chain {record['id']} routed to {route_token!r}; expected {PROT_CLA_TOKEN!r}"
            )
        prediction = infer_protein(model, tokenizer, esm_tokenizer, record["input"], args)
        pdb_residue_map = {
            str(residue["pos"]): residue["resi"]
            for residue in record["residues"]
        }
        result = {
            "record": record,
            "prediction": prediction,
            "pdb_residue_map": pdb_residue_map,
            "generation_question": generation_question,
            "generated_ids_shape": tuple(generated_ids.shape),
            "llm_output": generated_text,
            "final_special_token": effective_route_token,
            "raw_route_token": route_token,
            "route_fallback": route_fallback,
        }
        result["answer_text"] = format_protein_chain_answer(result)
        chain_results.append(result)

    prediction_jsonl = save_protein_prediction_jsonl(chain_results, args.protein_output_dir, item_index)
    viewers = build_protein_viewers(args.pdb_files, prediction_jsonl, args.protein_viewer_cdn)
    answer_text = summarize_protein_chain_answers(model, tokenizer, chain_results, args)
    return {
        "type": "protein_pdb",
        "answer_text": answer_text,
        "num_sequences": len(chain_results),
        "final_special_token": PROT_CLA_TOKEN,
        "generation_questions": [result["generation_question"] for result in chain_results],
        "llm_outputs": [result["llm_output"] for result in chain_results],
        "prediction_jsonl": prediction_jsonl,
        "chains": [
            {
                "id": result["record"]["id"],
                "pdb_id": result["record"]["pdb_id"],
                "pdb_file": result["record"]["pdb_file"],
                "chain": result["record"]["chain"],
                "sequence_length": result["record"]["sequence_length"],
                "positive_indices": result["prediction"]["positive_indices"],
                "probabilities": result["prediction"]["probabilities"],
                "threshold": result["prediction"]["threshold"],
                "pdb_residue_map": result["pdb_residue_map"],
                "answer_text": result["answer_text"],
                "generation_question": result["generation_question"],
                "llm_output": result["llm_output"],
                "final_special_token": result["final_special_token"],
                "raw_route_token": result["raw_route_token"],
                "route_fallback": result["route_fallback"],
            }
            for result in chain_results
        ],
        "visualizations": viewers,
        "visualization_html": "\n".join(viewer["html"] for viewer in viewers),
    }


def infer_property(model: S1Protein, tokenizer, predictor: nn.Module, generated_ids: torch.Tensor, stats: dict[str, Any], args):
    attention_mask = torch.ones_like(generated_ids)
    with torch.inference_mode():
        outputs = model.backbone.model(
            input_ids=generated_ids.to(args.device),
            attention_mask=attention_mask.to(args.device),
        )
        hidden_states = outputs[0]
        pooled = pool_question_hidden(
            tokenizer,
            hidden_states,
            generated_ids.to(args.device),
            attention_mask.to(args.device),
        )
        head_param = next(predictor.parameters())
        predictor_value = predictor(pooled.to(device=head_param.device, dtype=head_param.dtype))

    logit = float(predictor_value[0].detach().float().cpu().item())
    pred_class = int(logit >= 0.0)
    regression_value = denormalize_regression(logit, stats)
    prob_pos = float(torch.sigmoid(torch.tensor(logit)).item())
    return {
        "predictor_logit": logit,
        "classification": {
            "pred_class": pred_class,
            "prob": [1.0 - prob_pos, prob_pos],
        },
        "regression": {
            "value": regression_value,
            "normalized_value": logit,
            "normalization": stats.get("normalization", "none"),
            "label_mean": stats.get("mean"),
            "label_std": stats.get("std"),
            "label_min": stats.get("min"),
            "label_max": stats.get("max"),
            "label_range": stats.get("range"),
        },
    }


def infer_simplefold(
    model: S1Protein,
    tokenizer,
    simplefold_decoder: LazySimpleFoldDecoder,
    generated_ids: torch.Tensor,
    question: str,
    args,
    item_index: int | None = None,
):
    sequence = extract_simplefold_sequence(question)
    attention_mask = torch.ones_like(generated_ids)
    with torch.inference_mode():
        outputs = model.backbone.model(
            input_ids=generated_ids.to(args.device),
            attention_mask=attention_mask.to(args.device),
        )
        if getattr(outputs, "last_hidden_state", None) is not None:
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        else:
            raise RuntimeError("VLM prefill forward did not return hidden states")
        pooled_hidden = select_route_token_hidden(
            tokenizer,
            hidden_states,
            generated_ids.to(args.device),
            PROTEIN_FOLD_TOKEN,
            args.simplefold_route_token_policy,
        )

    simplefold_model = simplefold_decoder.get()
    if args.simplefold_sample_id:
        sample_id = args.simplefold_sample_id
    elif item_index is not None:
        sample_id = f"simplefold_{item_index:06d}"
    else:
        sample_id = "simplefold"
    workspace = None
    if args.simplefold_workspace:
        workspace = Path(args.simplefold_workspace) / safe_filename_stem(sample_id)

    output = simplefold_model.forward(
        sequence=sequence,
        seed=args.simplefold_seed,
        num_steps=args.simplefold_num_steps,
        tau=args.simplefold_tau,
        nsample_per_protein=1,
        sample_id=sample_id,
        workspace=workspace,
        pooled_text_hidden=pooled_hidden,
    )
    structure_path = save_simplefold_structure(
        simplefold_model,
        output,
        args.simplefold_output_dir,
        sample_id,
        item_index,
        args.simplefold_output_format,
    )
    return {
        "type": "simplefold",
        "structure_path": structure_path,
        "format": args.simplefold_output_format,
        "sample_id": sample_id,
        "sequence_length": len(sequence.replace(":", "")),
        "sequence": sequence,
        "record_id": getattr(output, "record_id", sample_id),
        "n_atoms": int(output.coordinates.shape[0]),
        "valid_atoms": int(output.pad_mask.sum().item()),
        "hidden_shape": list(pooled_hidden.shape),
        "num_steps": args.simplefold_num_steps,
        "tau": args.simplefold_tau,
        "seed": args.simplefold_seed,
    }


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_spectra_sdf(
    spec_model,
    result,
    output_dir: str | Path,
    item_index: int | None,
) -> str | None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = "sample" if item_index is None else f"sample_{item_index:06d}"
    path = output_path / f"{stem}.sdf"
    counter = 1
    while path.exists():
        path = output_path / f"{stem}_{counter}.sdf"
        counter += 1
    if spec_model.save_sdf(result, path):
        return str(path.resolve())
    return None


@torch.no_grad()
def extract_spectra_vision_hidden(
    model: S1Protein,
    processor,
    image_path: str | Path,
    modality: str,
    prompt: str,
    args,
) -> dict[str, Any]:
    from spec2mol_v2.vision_span import EXPECTED_VISION_SPAN, VISION_START_ID, slice_vision_span

    image = _load_rgb_image(image_path, target_size=SPECTRA_IMAGE_TARGET_SIZE)
    text = _apply_processor_template(processor, _build_spectra_messages([image], prompt))
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=False,
    )
    inputs = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    base = model.backbone.model if hasattr(model.backbone, "model") else model.backbone
    forward_kwargs: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "use_cache": False,
        "return_dict": True,
    }
    if "image_grid_thw" in inputs:
        forward_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
    for optional_key in ("mm_token_type_ids", "video_grid_thw", "position_ids"):
        if optional_key in inputs:
            forward_kwargs[optional_key] = inputs[optional_key]

    outputs = base(**forward_kwargs)
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    try:
        vision_span = slice_vision_span(hidden, inputs["input_ids"])
        ok = int(vision_span.shape[1]) == EXPECTED_VISION_SPAN
        error = None if ok else f"vision span length {vision_span.shape[1]} != {EXPECTED_VISION_SPAN}"
    except Exception as exc:  # noqa: BLE001
        vision_span = None
        ok = False
        error = repr(exc)

    return {
        "ok": ok,
        "modality": modality,
        "prompt": prompt,
        "vision_span": vision_span.detach().cpu() if vision_span is not None else None,
        "vision_span_shape": tuple(vision_span.shape) if vision_span is not None else None,
        "input_ids_shape": tuple(inputs["input_ids"].shape),
        "image_grid_thw": (
            inputs["image_grid_thw"].detach().cpu().tolist()
            if "image_grid_thw" in inputs
            else None
        ),
        "hidden_shape": tuple(hidden.shape),
        "vision_start_id": VISION_START_ID,
        "expected_vision_span": EXPECTED_VISION_SPAN,
        "target_size": list(SPECTRA_IMAGE_TARGET_SIZE),
        "error": error,
    }


def infer_spectra(
    model: S1Protein,
    processor,
    spec2mol_decoder: LazySpec2MolDecoder,
    spectra_context: dict[str, Any],
    question: str,
    args,
    item_index: int | None = None,
) -> dict[str, Any]:
    from spec2mol_v2.s1_omni_linear_predictor import DEFAULT_PROMPTS, MODALITIES

    prompts = dict(DEFAULT_PROMPTS)
    if args.ir_prompt:
        prompts["ir"] = args.ir_prompt
    if args.raman_prompt:
        prompts["raman"] = args.raman_prompt
    if args.uv_prompt:
        prompts["uv"] = args.uv_prompt

    present_images = spectra_context["present_images"]
    spans: dict[str, torch.Tensor] = {}
    span_metadata: dict[str, Any] = {}
    for modality in MODALITIES:
        if modality not in present_images:
            continue
        hidden = extract_spectra_vision_hidden(
            model,
            processor,
            present_images[modality],
            modality,
            prompts[modality],
            args,
        )
        span_metadata[modality] = {
            key: value
            for key, value in hidden.items()
            if key != "vision_span"
        }
        if not hidden["ok"]:
            raise RuntimeError(hidden["error"] or f"{modality} vision hidden extraction failed")
        spans[modality] = hidden["vision_span"]

    bank = spec2mol_decoder.get_bank()
    motif_out = bank.predict(spans)
    route_metadata = {
        "has_spectra_token": True,
        "image_order": spectra_context.get("image_order"),
        "prompt": question,
        "motif_fusion": motif_out["motif_fusion"],
        "predictor_modalities_used": motif_out["modalities_used"],
        "per_modality_prompt": {key: prompts[key] for key in spans},
        "hf_inline_modules": True,
    }
    with redirect_stdout(sys.stderr):
        spec_model = spec2mol_decoder.get_model()
        result = spec_model.generate_with_external_motif(
            bundle=spectra_context["bundle"],
            motif_onehot=motif_out["motif_onehot"],
            motif_probabilities=motif_out["probabilities"],
            route_metadata=route_metadata,
        )
    sdf_path = save_spectra_sdf(spec_model, result, args.spectra_output_dir, item_index)
    return {
        "type": "spectra",
        "sdf_path": sdf_path,
        "smiles": result.smiles,
        "atom_count": int(result.atom_count),
        "motif_onehot": result.motif_onehot,
        "motif_probabilities": result.motif_probabilities,
        "active_motifs": result.metadata.get("motif_active"),
        "modalities_used": motif_out["modalities_used"],
        "jdx_files": spectra_context["jdx_files"],
        "span_metadata": to_jsonable(span_metadata),
        "route_metadata": to_jsonable(route_metadata),
        "motif_prediction": to_jsonable(
            {
                key: value
                for key, value in motif_out.items()
                if key != "per_modality"
            }
        ),
        "metadata": to_jsonable(result.metadata),
    }


def infer_once(
    model,
    tokenizer,
    processor,
    esm_tokenizer,
    predictor,
    image_decoder: LazyImageDecoder,
    simplefold_decoder: LazySimpleFoldDecoder,
    spec2mol_decoder: LazySpec2MolDecoder,
    stats,
    question: str,
    args,
    item_index: int | None = None,
):
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.pdb_files:
            prediction = infer_protein_from_pdb(
                model,
                tokenizer,
                processor,
                esm_tokenizer,
                question,
                args,
                item_index=item_index,
            )
            return {
                "question": question,
                "generation_question": prediction["generation_questions"],
                "llm_output": prediction["llm_outputs"],
                "final_special_token": prediction["final_special_token"],
                "selected_by": "protein",
                "prediction": prediction,
                "final_text": prediction["answer_text"],
            }

        spectra_context = None
        if args.jdx_files:
            temp_dir = tempfile.TemporaryDirectory(prefix="s1_omni_spectra_")
            spectra_context = prepare_spectra_context(args, Path(temp_dir.name))

        with torch.inference_mode():
            generation_question, _, generated_ids, generated_text, route_token, condition_drop_idx = generate_route(
                model,
                tokenizer,
                processor,
                question,
                args,
                spectra_context=spectra_context,
            )

        prediction: dict[str, Any] | None = None
        replacement = ""
        selected_by = "unknown"
        if route_token == PROT_CLA_TOKEN:
            selected_by = "protein"
            prediction = infer_protein(model, tokenizer, esm_tokenizer, question, args)
            replacement = json.dumps(prediction["positive_indices"], ensure_ascii=False)
        elif route_token == PROTEIN_FOLD_TOKEN:
            selected_by = "simplefold"
            prediction = infer_simplefold(
                model,
                tokenizer,
                simplefold_decoder,
                generated_ids,
                question,
                args,
                item_index=item_index,
            )
            replacement = prediction["structure_path"]
        elif route_token in {LINEAR_CLA_TOKEN, LINEAR_PRE_TOKEN}:
            property_prediction = infer_property(model, tokenizer, predictor, generated_ids, stats, args)
            prediction = property_prediction
            if route_token == LINEAR_CLA_TOKEN:
                selected_by = "classification"
                replacement = str(property_prediction["classification"]["pred_class"])
                if replacement == "0":
                    replacement = "No"
                else:
                    replacement = "Yes"
            else:
                selected_by = "regression"
                replacement = str(property_prediction["regression"]["value"])
        elif route_token in {IMAGE_GEN_TOKEN, IMAGE_EDIT_TOKEN}:
            source_image = load_pil_image(args.image_input)
            decoder = image_decoder.get()
            image, image_task = decoder.infer_from_generated(
                route_token=route_token,
                generated_ids=generated_ids,
                condition_drop_idx=condition_drop_idx,
                source_image=source_image,
                args=args,
            )
            image_path = save_generated_image(image, args.image_output_dir, item_index, image_task)
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
        elif route_token == SPECTRA_TOKEN or args.force_spectra:
            if spectra_context is None:
                raise ValueError("--jdx_files is required when the selected route is <spectra_st>")
            selected_by = "spectra"
            prediction = infer_spectra(
                model,
                processor,
                spec2mol_decoder,
                spectra_context,
                question,
                args,
                item_index=item_index,
            )
            replacement = prediction.get("sdf_path") or prediction.get("smiles") or ""

        if prediction is None:
            final_text = generated_text
        elif selected_by == "spectra" and args.force_spectra and route_token != SPECTRA_TOKEN and SPECTRA_TOKEN not in generated_text:
            final_text = replacement
        else:
            final_text = replace_last_route_token(generated_text, route_token, replacement)
        return {
            "question": question,
            "generation_question": generation_question,
            "llm_output": generated_text,
            "final_special_token": route_token,
            "selected_by": selected_by,
            "prediction": prediction,
            "final_text": final_text,
        }
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def write_results(results: list[dict[str, Any]], output_file: str | None):
    if output_file is None:
        payload: dict[str, Any] | list[dict[str, Any]]
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if args.image_input and args.jdx_files:
        raise ValueError("--image_input and --jdx_files cannot be used in the same request")
    if args.pdb_files and (args.image_input or args.jdx_files):
        raise ValueError("--pdb_files cannot be used with --image_input or --jdx_files")
    stats = load_regression_stats(args.stats_path)
    (
        model,
        tokenizer,
        processor,
        esm_tokenizer,
        predictor,
        image_decoder,
        simplefold_decoder,
        spec2mol_decoder,
    ) = load_dual_model(args)

    if args.input_file:
        questions = read_questions(args.input_file)
        results = [
            infer_once(
                model,
                tokenizer,
                processor,
                esm_tokenizer,
                predictor,
                image_decoder,
                simplefold_decoder,
                spec2mol_decoder,
                stats,
                question,
                args,
                item_index=index,
            )
            for index, question in enumerate(questions)
        ]
        write_results(results, args.output_file)
        return

    if args.question is not None or args.question_file is not None:
        result = infer_once(
            model,
            tokenizer,
            processor,
            esm_tokenizer,
            predictor,
            image_decoder,
            simplefold_decoder,
            spec2mol_decoder,
            stats,
            read_question(args),
            args,
        )
        write_results([result], args.output_file)
        return

    print("S1 dual inference ready. 输入问题后回车，输入 q 退出。")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            break
        if not question or question.lower() == "q":
            break
        result = infer_once(
            model,
            tokenizer,
            processor,
            esm_tokenizer,
            predictor,
            image_decoder,
            simplefold_decoder,
            spec2mol_decoder,
            stats,
            question,
            args,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
