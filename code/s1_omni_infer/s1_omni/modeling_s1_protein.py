import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import ModelOutput

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except ImportError:
    safe_load_file = None
    safe_save_file = None

S1_PROTEIN_CONFIG_NAME = "s1_protein_config.json"
S1_PROTEIN_VLM_DIR_NAME = "vlm"
S1_PROTEIN_DECODER_WEIGHTS_NAME = "protein_decoder.pt"
S1_PROTEIN_DECODER_FORMAT = "s1_protein_decoder"
S1_PROTEIN_MODULE_PREFIXES = (
    "protein_head.",
    "esm_query_projection.",
    "qwen_key_projection.",
    "qwen_value_projection.",
    "fusion_layers.",
    "cross_attention.",
    "fusion_norm.",
)
ESM2_SAVED_DIR_NAME = "esm2"
ESM2_SAVED_WEIGHTS_NAME = "model.safetensors"
ESM2_SAVED_CONFIG_NAME = "esm2_config.json"


@dataclass
class S1ProteinOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    pred_bits: torch.Tensor | None = None
    backbone_hidden_states: tuple[torch.Tensor, ...] | None = None
    backbone_attentions: tuple[torch.Tensor, ...] | None = None


class ProteinCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        num_attention_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(fusion_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, fusion_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        residue_hidden: torch.Tensor,
        qwen_key: torch.Tensor,
        qwen_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.attention_norm(residue_hidden)
        attn_output, _ = self.cross_attention(
            query=attn_input,
            key=qwen_key,
            value=qwen_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        residue_hidden = residue_hidden + self.attention_dropout(attn_output)

        ffn_input = self.ffn_norm(residue_hidden)
        residue_hidden = residue_hidden + self.ffn_dropout(self.ffn(ffn_input))
        return residue_hidden * residue_mask.unsqueeze(-1).to(residue_hidden.dtype)


class S1Protein(nn.Module):
    def __init__(
        self,
        backbone,
        hidden_size: int,
        output_size: int = 1,
        head_hidden_size: int | None = None,
        head_dropout: float = 0.1,
        use_esm2: bool = True,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        esm_fusion_dim: int = 512,
        esm_num_attention_heads: int = 8,
        esm_fusion_num_layers: int = 2,
        esm_fusion_ffn_dim: int | None = None,
        esm_dtype=None,
        esm_unfreeze_last_n_layers: int = 0,
        esm_unfreeze_pooler: bool = False,
        esm_unfreeze_final_layer_norm: bool = True,
        esm_lr_multiplier: float = 0.1,
        positive_loss_weight: float = 50.0,
        protein_loss_type: str = "bce",
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        asl_eps: float = 1e-8,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.head_hidden_size = head_hidden_size or max(256, hidden_size // 4)
        self.head_dropout = head_dropout
        self.use_esm2 = use_esm2
        self.esm_model_name = esm_model_name
        self.esm_fusion_dim = esm_fusion_dim
        self.esm_num_attention_heads = esm_num_attention_heads
        self.esm_fusion_num_layers = esm_fusion_num_layers
        self.esm_fusion_ffn_dim = esm_fusion_ffn_dim or esm_fusion_dim * 4
        self.positive_loss_weight = positive_loss_weight
        if esm_unfreeze_last_n_layers < 0:
            raise ValueError("esm_unfreeze_last_n_layers must be >= 0")
        self.esm_unfreeze_last_n_layers = int(esm_unfreeze_last_n_layers)
        self.esm_unfreeze_pooler = bool(esm_unfreeze_pooler)
        self.esm_unfreeze_final_layer_norm = bool(esm_unfreeze_final_layer_norm)
        self.esm_lr_multiplier = float(esm_lr_multiplier)
        self.protein_loss_type = protein_loss_type.lower()
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_clip = asl_clip
        self.asl_eps = asl_eps
        self.tokenizer = None
        self.backbone_frozen = False
        if self.protein_loss_type not in {"bce", "asl"}:
            raise ValueError("protein_loss_type must be one of: bce, asl")

        head_input_size = esm_fusion_dim if use_esm2 else hidden_size
        self.protein_head = nn.Sequential(
            nn.LayerNorm(head_input_size),
            nn.Linear(head_input_size, self.head_hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(self.head_hidden_size, output_size),
        )
        self.esm_model = None
        if self.use_esm2:
            self.esm_model = self._load_frozen_esm2(
                self._resolve_esm_model_source(esm_model_name),
                esm_dtype,
            )
            esm_hidden_size = int(getattr(self.esm_model.config, "hidden_size"))
            self.esm_query_projection = nn.Linear(esm_hidden_size, esm_fusion_dim)
            self.qwen_key_projection = nn.Linear(hidden_size, esm_fusion_dim)
            self.qwen_value_projection = nn.Linear(hidden_size, esm_fusion_dim)
            if esm_fusion_num_layers < 1:
                raise ValueError("esm_fusion_num_layers must be >= 1")
            self.fusion_layers = nn.ModuleList(
                [
                    ProteinCrossAttentionBlock(
                        fusion_dim=esm_fusion_dim,
                        num_attention_heads=esm_num_attention_heads,
                        ffn_dim=self.esm_fusion_ffn_dim,
                        dropout=head_dropout,
                    )
                    for _ in range(esm_fusion_num_layers)
                ]
            )
            self.fusion_norm = nn.LayerNorm(esm_fusion_dim)
        self._init_head_module(self.protein_head)
        if self.use_esm2:
            self._init_head_module(self.esm_query_projection)
            self._init_head_module(self.qwen_key_projection)
            self._init_head_module(self.qwen_value_projection)
            self._init_head_module(self.fusion_layers)
            self._init_head_module(self.fusion_norm)

        for p in self.backbone.parameters():
            p.requires_grad = False

    @staticmethod
    def _load_frozen_esm2(model_name_or_path: str, esm_dtype=None):
        load_kwargs = {}
        if esm_dtype is not None:
            load_kwargs["torch_dtype"] = esm_dtype
        try:
            esm_model = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
        except TypeError:
            if esm_dtype is not None:
                load_kwargs = {"dtype": esm_dtype}
                esm_model = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
            else:
                raise
        esm_model.requires_grad_(False)
        esm_model.eval()
        return esm_model

    @staticmethod
    def _resolve_esm_model_source(model_name_or_path: str):
        if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
            esm_dir = os.path.join(model_name_or_path, ESM2_SAVED_DIR_NAME)
            weights_path = os.path.join(esm_dir, ESM2_SAVED_WEIGHTS_NAME)
            config_path = os.path.join(esm_dir, "config.json")
            if (
                os.path.exists(weights_path)
                and os.path.exists(config_path)
                and S1Protein._esm2_weights_look_complete(weights_path)
            ):
                return esm_dir
        return model_name_or_path

    @staticmethod
    def _esm2_weights_look_complete(weights_path: str) -> bool:
        if not os.path.exists(weights_path):
            return False
        if weights_path.endswith(".safetensors"):
            if safe_load_file is None:
                return False
            try:
                from safetensors.torch import safe_open

                with safe_open(weights_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        shape = tuple(f.get_slice(key).get_shape())
                        if not shape or any(dim == 0 for dim in shape):
                            return False
                return True
            except Exception:
                return False

        try:
            state_dict = S1Protein._load_checkpoint_file(weights_path)
        except Exception:
            return False
        return bool(state_dict) and all(
            tuple(value.shape) and not any(dim == 0 for dim in tuple(value.shape))
            for value in state_dict.values()
            if torch.is_tensor(value)
        )

    @staticmethod
    def resolve_esm_tokenizer_source(model_name_or_path: str, esm_model_name: str):
        if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
            esm_dir = os.path.join(model_name_or_path, ESM2_SAVED_DIR_NAME)
            tokenizer_config = os.path.join(esm_dir, "tokenizer_config.json")
            vocab_file = os.path.join(esm_dir, "vocab.txt")
            if os.path.exists(tokenizer_config) or os.path.exists(vocab_file):
                return esm_dir
        return esm_model_name

    def _get_esm_encoder_layers(self) -> list[nn.Module]:
        """Return the list of transformer layers in the ESM2 encoder."""
        if self.esm_model is None:
            return []
        encoder = getattr(self.esm_model, "encoder", None)
        if encoder is None:
            return []
        layers = getattr(encoder, "layer", None)
        if layers is None:
            return []
        return list(layers)

    def _apply_esm_unfreeze_config(self) -> None:
        """Apply the configured ESM2 partial-unfreeze policy.

        - Freeze everything first.
        - Unfreeze the last ``esm_unfreeze_last_n_layers`` transformer layers
          in their entirety.
        - Optionally unfreeze the pooler and ``emb_layer_norm_after``.

        Always keep ``esm_model.eval()`` so dropout inside ESM2 stays off.
        """
        if self.esm_model is None:
            return
        self.esm_model.requires_grad_(False)
        self.esm_model.eval()

        layers = self._get_esm_encoder_layers()
        n_total = len(layers)
        n_unfreeze = min(self.esm_unfreeze_last_n_layers, n_total)
        for layer in layers[n_total - n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True

        if self.esm_unfreeze_final_layer_norm:
            encoder = getattr(self.esm_model, "encoder", None)
            for attr in ("emb_layer_norm_after", "layer_norm"):
                module = getattr(encoder, attr, None) if encoder is not None else None
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = True

        if self.esm_unfreeze_pooler:
            pooler = getattr(self.esm_model, "pooler", None)
            if pooler is not None:
                for p in pooler.parameters():
                    p.requires_grad = True

    def trainable_esm_param_count(self) -> int:
        """Number of trainable parameters inside the ESM2 submodule."""
        if self.esm_model is None:
            return 0
        return sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)

    @property
    def supports_gradient_checkpointing(self):
        if self.backbone_frozen:
            return False
        return hasattr(self.backbone, "supports_gradient_checkpointing") or hasattr(
            self.backbone, "gradient_checkpointing_enable"
        )

    @property
    def is_gradient_checkpointing(self):
        if hasattr(self.backbone, "is_gradient_checkpointing"):
            return self.backbone.is_gradient_checkpointing
        return False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if self.backbone_frozen:
            return None
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            return self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        raise AttributeError(
            f"{self.__class__.__name__} does not support gradient checkpointing"
        )

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            return self.backbone.gradient_checkpointing_disable()
        raise AttributeError(
            f"{self.__class__.__name__} does not support gradient checkpointing"
        )

    def enable_input_require_grads(self):
        if hasattr(self.backbone, "enable_input_require_grads"):
            return self.backbone.enable_input_require_grads()
        return None

    def disable_input_require_grads(self):
        if hasattr(self.backbone, "disable_input_require_grads"):
            return self.backbone.disable_input_require_grads()
        return None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        dtype=None,
        use_esm2: Optional[bool] = None,
        esm_model_name: Optional[str] = None,
        esm_fusion_dim: Optional[int] = None,
        esm_num_attention_heads: Optional[int] = None,
        esm_fusion_num_layers: Optional[int] = None,
        esm_fusion_ffn_dim: Optional[int] = None,
        esm_unfreeze_last_n_layers: Optional[int] = None,
        esm_unfreeze_pooler: Optional[bool] = None,
        esm_unfreeze_final_layer_norm: Optional[bool] = None,
        esm_lr_multiplier: Optional[float] = None,
        positive_loss_weight: Optional[float] = None,
        protein_loss_type: Optional[str] = None,
        asl_gamma_pos: Optional[float] = None,
        asl_gamma_neg: Optional[float] = None,
        asl_clip: Optional[float] = None,
        asl_eps: Optional[float] = None,
        **kwargs,
    ):
        vlm_model_name_or_path = cls._resolve_vlm_model_source(model_name_or_path)
        backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=dtype,
            **kwargs,
        )
        hidden_size = backbone.config.text_config.hidden_size
        saved_config = cls._load_s1_protein_config(model_name_or_path)
        init_kwargs = {
            "use_esm2": saved_config.get("use_esm2", True),
            "esm_model_name": saved_config.get("esm_model_name", "facebook/esm2_t33_650M_UR50D"),
            "esm_fusion_dim": saved_config.get("esm_fusion_dim", 512),
            "esm_num_attention_heads": saved_config.get("esm_num_attention_heads", 8),
            "esm_fusion_num_layers": saved_config.get(
                "esm_fusion_num_layers",
                1 if saved_config else 2,
            ),
            "esm_fusion_ffn_dim": saved_config.get("esm_fusion_ffn_dim", None),
            "esm_unfreeze_last_n_layers": saved_config.get("esm_unfreeze_last_n_layers", 0),
            "esm_unfreeze_pooler": saved_config.get("esm_unfreeze_pooler", False),
            "esm_unfreeze_final_layer_norm": saved_config.get("esm_unfreeze_final_layer_norm", True),
            "esm_lr_multiplier": saved_config.get("esm_lr_multiplier", 0.1),
            "positive_loss_weight": saved_config.get("positive_loss_weight", 50.0),
            "protein_loss_type": saved_config.get("protein_loss_type", "bce"),
            "asl_gamma_pos": saved_config.get("asl_gamma_pos", 0.0),
            "asl_gamma_neg": saved_config.get("asl_gamma_neg", 4.0),
            "asl_clip": saved_config.get("asl_clip", 0.05),
            "asl_eps": saved_config.get("asl_eps", 1e-8),
            "esm_dtype": dtype,
        }
        if use_esm2 is not None:
            init_kwargs["use_esm2"] = use_esm2
        if esm_model_name is not None:
            init_kwargs["esm_model_name"] = esm_model_name
        if esm_fusion_dim is not None:
            init_kwargs["esm_fusion_dim"] = esm_fusion_dim
        if esm_num_attention_heads is not None:
            init_kwargs["esm_num_attention_heads"] = esm_num_attention_heads
        if esm_fusion_num_layers is not None:
            init_kwargs["esm_fusion_num_layers"] = esm_fusion_num_layers
        if esm_fusion_ffn_dim is not None:
            init_kwargs["esm_fusion_ffn_dim"] = esm_fusion_ffn_dim
        if esm_unfreeze_last_n_layers is not None:
            init_kwargs["esm_unfreeze_last_n_layers"] = esm_unfreeze_last_n_layers
        if esm_unfreeze_pooler is not None:
            init_kwargs["esm_unfreeze_pooler"] = esm_unfreeze_pooler
        if esm_unfreeze_final_layer_norm is not None:
            init_kwargs["esm_unfreeze_final_layer_norm"] = esm_unfreeze_final_layer_norm
        if esm_lr_multiplier is not None:
            init_kwargs["esm_lr_multiplier"] = esm_lr_multiplier
        if positive_loss_weight is not None:
            init_kwargs["positive_loss_weight"] = positive_loss_weight
        if protein_loss_type is not None:
            init_kwargs["protein_loss_type"] = protein_loss_type
        if asl_gamma_pos is not None:
            init_kwargs["asl_gamma_pos"] = asl_gamma_pos
        if asl_gamma_neg is not None:
            init_kwargs["asl_gamma_neg"] = asl_gamma_neg
        if asl_clip is not None:
            init_kwargs["asl_clip"] = asl_clip
        if asl_eps is not None:
            init_kwargs["asl_eps"] = asl_eps
        bundled_esm_dir = os.path.join(model_name_or_path, ESM2_SAVED_DIR_NAME)
        bundled_esm_weights = os.path.join(bundled_esm_dir, ESM2_SAVED_WEIGHTS_NAME)
        bundled_esm_config = os.path.join(bundled_esm_dir, "config.json")
        if (
            init_kwargs["use_esm2"]
            and os.path.exists(bundled_esm_weights)
            and os.path.exists(bundled_esm_config)
            and cls._esm2_weights_look_complete(bundled_esm_weights)
        ):
            init_kwargs["esm_model_name"] = bundled_esm_dir
        model = cls(backbone=backbone, hidden_size=hidden_size, **init_kwargs)
        target_dtype = next(backbone.parameters()).dtype
        model.to(dtype=target_dtype)
        if model.esm_model is not None:
            model.esm_model.to(dtype=target_dtype)
            require_bundled_esm = bool(saved_config) and (
                int(init_kwargs.get("esm_unfreeze_last_n_layers", 0) or 0) > 0
                or bool(init_kwargs.get("esm_unfreeze_pooler", False))
                or bool(init_kwargs.get("esm_unfreeze_final_layer_norm", False))
            )
            model._load_esm2_from_pretrained(
                model_name_or_path,
                require_bundled=require_bundled_esm,
            )
            model._apply_esm_unfreeze_config()
        model._load_head_from_pretrained(model_name_or_path)
        model.config = backbone.config
        model.processor_name_or_path = model_name_or_path
        return model

    @staticmethod
    def _resolve_vlm_model_source(model_name_or_path: str):
        if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
            vlm_dir = os.path.join(model_name_or_path, S1_PROTEIN_VLM_DIR_NAME)
            if os.path.exists(os.path.join(vlm_dir, "config.json")):
                return vlm_dir
        return model_name_or_path

    @staticmethod
    def _load_s1_protein_config(model_name_or_path: str):
        config_path = os.path.join(model_name_or_path, S1_PROTEIN_CONFIG_NAME)
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "disable_input_require_grads"):
            self.backbone.disable_input_require_grads()
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        self.backbone_frozen = True

    def unfreeze_llm_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        language_model = getattr(getattr(self.backbone, "model", None), "language_model", None)
        if language_model is None:
            language_model = getattr(self.backbone, "language_model", None)
        if language_model is None:
            raise AttributeError("S1Protein backbone has no language_model module to unfreeze")
        for p in language_model.parameters():
            p.requires_grad = True

        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        if self.esm_model is not None:
            self._apply_esm_unfreeze_config()
        self.backbone_frozen = False

    @staticmethod
    def _init_head_module(module: nn.Module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    @staticmethod
    def _load_checkpoint_file(path: str):
        if path.endswith(".safetensors"):
            if safe_load_file is None:
                raise ImportError("safetensors is required to load safetensors checkpoints")
            return safe_load_file(path, device="cpu")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"]
        return checkpoint

    @classmethod
    def _load_state_dict_files(cls, model_name_or_path: str):
        if not os.path.isdir(model_name_or_path):
            return []

        files = []
        decoder_path = os.path.join(model_name_or_path, S1_PROTEIN_DECODER_WEIGHTS_NAME)
        if os.path.exists(decoder_path):
            files.append(decoder_path)
            return files

        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = os.path.join(model_name_or_path, index_name)
            if not os.path.exists(index_path):
                continue
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            head_files = {
                filename
                for key, filename in index.get("weight_map", {}).items()
                if key.startswith(S1_PROTEIN_MODULE_PREFIXES)
            }
            files.extend(os.path.join(model_name_or_path, filename) for filename in sorted(head_files))
            return files

        for filename in ("model.safetensors", "pytorch_model.bin"):
            path = os.path.join(model_name_or_path, filename)
            if os.path.exists(path):
                files.append(path)
        return files

    def _load_esm2_from_pretrained(
        self,
        model_name_or_path: str,
        require_bundled: bool = False,
    ) -> bool:
        """Load bundled ESM2 weights from ``<model_name_or_path>/esm2/`` if present.

        Returns True if bundled weights were found and loaded, False otherwise.

        When ``require_bundled`` is True and the bundled file is missing or
        clearly corrupted (e.g. zero-shape shards from a ZeRO-3 save), this
        raises a RuntimeError instead of silently falling back to the
        ``esm_model_name`` external weights. This protects the inference path
        from accidentally using the wrong ESM2 weights when the training
        process was supposed to update them.
        """
        if self.esm_model is None:
            return False
        if not isinstance(model_name_or_path, str) or not os.path.isdir(model_name_or_path):
            if require_bundled:
                raise RuntimeError(
                    f"require_bundled=True but checkpoint path is invalid: {model_name_or_path}"
                )
            return False
        esm_dir = os.path.join(model_name_or_path, ESM2_SAVED_DIR_NAME)
        weights_path = os.path.join(esm_dir, ESM2_SAVED_WEIGHTS_NAME)
        if not os.path.exists(weights_path):
            if require_bundled:
                raise RuntimeError(
                    f"require_bundled=True but {weights_path} does not exist. "
                    f"Cannot fall back to esm_model_name={self.esm_model_name!r}."
                )
            return False
        state_dict = self._load_checkpoint_file(weights_path)
        current_state_dict = self.esm_model.state_dict()
        compatible = {}
        rejected_zero_shard = 0
        rejected_shape_mismatch = 0
        rejected_missing_key = 0
        for key, value in state_dict.items():
            if key not in current_state_dict:
                rejected_missing_key += 1
                continue
            target_shape = tuple(current_state_dict[key].shape)
            value_shape = tuple(value.shape)
            if value_shape == target_shape and not any(s == 0 for s in value_shape):
                compatible[key] = value
            elif value_shape == target_shape:
                rejected_zero_shard += 1
            else:
                rejected_shape_mismatch += 1
        if require_bundled and (
            rejected_missing_key > 0
            or rejected_shape_mismatch > 0
            or rejected_zero_shard > 0
            or len(compatible) != len(current_state_dict)
        ):
            raise RuntimeError(
                f"require_bundled=True but {weights_path} is not a complete ESM2 "
                f"checkpoint: compatible={len(compatible)}/{len(current_state_dict)}, "
                f"rejected missing_key={rejected_missing_key}, "
                f"shape_mismatch={rejected_shape_mismatch}, "
                f"zero_shard={rejected_zero_shard}. "
                "This usually means ESM2 was saved from DeepSpeed ZeRO-3 as "
                "empty per-rank shards. Re-save ESM2 with full parameter gathering "
                "or rerun training with the fixed save code."
            )
        if not compatible:
            if require_bundled:
                raise RuntimeError(
                    f"require_bundled=True but {weights_path} contained no compatible "
                    f"tensors for ESM2 (file looks corrupted, possibly ZeRO-3 sharded)."
                )
            print(
                f"[S1Protein] bundled ESM2 weights at {weights_path} had no compatible "
                f"tensors; falling back to esm_model_name={self.esm_model_name!r}."
            )
            return False
        result = self.esm_model.load_state_dict(compatible, strict=False)
        print(
            f"[S1Protein] loaded bundled ESM2 from {weights_path}: "
            f"{len(compatible)} tensors applied; "
            f"rejected missing_key={rejected_missing_key}, "
            f"shape_mismatch={rejected_shape_mismatch}, "
            f"zero_shard={rejected_zero_shard}; "
            f"unexpected_keys={len(result.unexpected_keys)}, "
            f"missing_keys={len(result.missing_keys)}."
        )
        return True

    def _save_esm2_pretrained(
        self,
        save_directory: str,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """Dump the full ESM2 state_dict to ``<save_directory>/esm2/``.

        Prefer a caller-provided full ``state_dict`` when available (for
        example the ZeRO-3 gathered state dict that HuggingFace Trainer passes
        into ``save_pretrained`` during checkpoint save). Falling back to
        ``self.esm_model.state_dict()`` under ZeRO-3 can otherwise surface
        empty per-rank shards such as ``(0, hidden_size)`` embeddings.
        """
        if self.esm_model is None:
            return
        esm_dir = os.path.join(save_directory, ESM2_SAVED_DIR_NAME)
        os.makedirs(esm_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self._gather_full_esm2_state_dict()
        if state_dict is None:
            return
        if not state_dict:
            return
        bad_tensors = [
            key
            for key, tensor in state_dict.items()
            if tensor.numel() == 0 or any(dim == 0 for dim in tuple(tensor.shape))
        ]
        if bad_tensors:
            preview = ", ".join(bad_tensors[:5])
            raise RuntimeError(
                "failed to gather full ESM2 state_dict before saving; "
                f"{len(bad_tensors)} tensors still have zero shape/numel, e.g. {preview}"
            )
        weights_path = os.path.join(esm_dir, ESM2_SAVED_WEIGHTS_NAME)
        if safe_save_file is not None:
            safe_save_file(state_dict, weights_path)
        else:
            torch.save(state_dict, weights_path)
        config = getattr(self.esm_model, "config", None)
        if config is not None:
            try:
                config_dict = config.to_dict()
            except Exception:
                config_dict = dict(getattr(config, "__dict__", {}))
            hf_config_path = os.path.join(esm_dir, "config.json")
            with open(hf_config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2, default=str)
            config_path = os.path.join(esm_dir, ESM2_SAVED_CONFIG_NAME)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2, default=str)
        self._save_esm2_tokenizer(esm_dir)

    def _save_esm2_tokenizer(self, esm_dir: str) -> None:
        tokenizer_source = self.resolve_esm_tokenizer_source(
            self.processor_name_or_path,
            self.esm_model_name,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            tokenizer.save_pretrained(esm_dir)
        except Exception as exc:
            print(f"[S1Protein] failed to save ESM2 tokenizer from {tokenizer_source!r}: {exc}")

    def _gather_full_esm2_state_dict(self) -> dict[str, torch.Tensor] | None:
        """Return the full (unsharded) ESM2 state_dict on the gathering rank.

        - On non-distributed setups, returns ``self.esm_model.state_dict()``.
        - Under DeepSpeed ZeRO-3, gathers all parameters on rank 0 and returns
          the rank-0 full state_dict; other ranks return an empty dict.
        - Returns None if the ESM2 module is absent.
        """
        if self.esm_model is None:
            return None

        esm_params = list(self.esm_model.parameters())
        if not esm_params:
            return None

        GatheredParameters = None
        try:
            from deepspeed.zero import GatheredParameters  # type: ignore
        except Exception:
            GatheredParameters = None

        if GatheredParameters is not None and self._is_distributed():
            rank = self._get_rank()
            with GatheredParameters(esm_params, modifier_rank=0):
                if rank != 0:
                    return {}
                return {
                    key: tensor.detach().contiguous().cpu()
                    for key, tensor in self.esm_model.state_dict().items()
                    if torch.is_tensor(tensor)
                }

        return {
            key: tensor.detach().contiguous().cpu()
            for key, tensor in self.esm_model.state_dict().items()
        }

    @staticmethod
    def _is_distributed() -> bool:
        return bool(
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

    @staticmethod
    def _get_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return 0

    def _load_head_from_pretrained(self, model_name_or_path: str):
        module_state_dict = {}
        for path in self._load_state_dict_files(model_name_or_path):
            state_dict = self._extract_state_dict(self._load_checkpoint_file(path))
            for key, value in state_dict.items():
                if key.startswith(S1_PROTEIN_MODULE_PREFIXES):
                    module_state_dict[key] = value

        if not module_state_dict:
            return

        for key, value in list(module_state_dict.items()):
            if key.startswith("cross_attention."):
                mapped_key = "fusion_layers.0.cross_attention." + key.removeprefix("cross_attention.")
                module_state_dict.setdefault(mapped_key, value)

        for module_name in (
            "protein_head",
            "esm_query_projection",
            "qwen_key_projection",
            "qwen_value_projection",
            "fusion_layers",
            "fusion_norm",
        ):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            prefix = f"{module_name}."
            current_state_dict = module.state_dict()
            compatible_state_dict = {
                key.removeprefix(prefix): value
                for key, value in module_state_dict.items()
                if key.startswith(prefix)
                and key.removeprefix(prefix) in current_state_dict
                and tuple(value.shape) == tuple(current_state_dict[key.removeprefix(prefix)].shape)
            }
            if compatible_state_dict:
                module.load_state_dict(compatible_state_dict, strict=False)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _pool_last_token(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        if attention_mask is None:
            return hidden_states[:, -1]
        lengths = attention_mask.long().sum(dim=-1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def _select_protein_hidden(
        self,
        hidden_states: torch.Tensor,
        protein_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        protein_token_mask = protein_token_mask.to(device=hidden_states.device, dtype=torch.bool)
        residue_counts = protein_token_mask.long().sum(dim=-1)
        max_residue_count = int(residue_counts.max().item()) if residue_counts.numel() else 0
        if max_residue_count == 0:
            raise ValueError("protein_token_mask does not select any residue tokens")

        batch_size, _, hidden_size = hidden_states.shape
        protein_hidden = hidden_states.new_zeros(batch_size, max_residue_count, hidden_size)
        residue_positions = torch.arange(max_residue_count, device=hidden_states.device).unsqueeze(0)
        residue_mask = residue_positions < residue_counts.unsqueeze(1)
        protein_hidden[residue_mask] = hidden_states[protein_token_mask]
        return protein_hidden, residue_mask

    def _weighted_residue_bce(
        self,
        logit_values: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pos_weight = torch.tensor(
            float(self.positive_loss_weight),
            device=logit_values.device,
            dtype=torch.float32,
        )
        per_residue_loss = F.binary_cross_entropy_with_logits(
            logit_values.float(),
            target.float(),
            pos_weight=pos_weight,
            reduction="none",
        )
        return (per_residue_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)

    def _asymmetric_residue_loss(
        self,
        logit_values: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = target.float()
        probability = torch.sigmoid(logit_values.float())
        positive_probability = probability.clamp(min=self.asl_eps, max=1.0)
        negative_probability = (1.0 - probability).clamp(min=self.asl_eps, max=1.0)
        if self.asl_clip > 0:
            negative_probability = (negative_probability + self.asl_clip).clamp(max=1.0)

        positive_loss = target * torch.log(positive_probability)
        if self.positive_loss_weight != 1.0:
            positive_loss = positive_loss * float(self.positive_loss_weight)
        negative_loss = (1.0 - target) * torch.log(negative_probability.clamp(min=self.asl_eps))
        per_residue_loss = -(positive_loss + negative_loss)

        if self.asl_gamma_pos > 0 or self.asl_gamma_neg > 0:
            pt = target * positive_probability + (1.0 - target) * negative_probability
            gamma = target * float(self.asl_gamma_pos) + (1.0 - target) * float(self.asl_gamma_neg)
            per_residue_loss = per_residue_loss * torch.pow((1.0 - pt).clamp(min=0.0), gamma)

        return (per_residue_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)

    def _protein_residue_loss(
        self,
        logit_values: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.protein_loss_type == "asl":
            return self._asymmetric_residue_loss(logit_values, target, valid_mask)
        return self._weighted_residue_bce(logit_values, target, valid_mask)

    def _build_residue_mask_from_esm_attention(self, esm_attention_mask: torch.Tensor):
        residue_counts = esm_attention_mask.long().sum(dim=-1).sub(2).clamp(min=0)
        max_residue_count = int(residue_counts.max().item()) if residue_counts.numel() else 0
        if max_residue_count == 0:
            raise ValueError("esm_attention_mask does not contain any residue tokens")
        positions = torch.arange(max_residue_count, device=esm_attention_mask.device).unsqueeze(0)
        residue_mask = positions < residue_counts.unsqueeze(1)
        return residue_mask, max_residue_count

    def _get_esm_max_positions(self):
        max_positions = getattr(getattr(self.esm_model, "config", None), "max_position_embeddings", None)
        return int(max_positions) if max_positions is not None else None

    def _compute_chunked_esm_hidden(
        self,
        esm_input_ids: torch.Tensor,
        esm_attention_mask: torch.Tensor,
        target_dtype: torch.dtype,
        max_positions: int,
        max_residue_count: int,
        residue_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_residues_per_chunk = max_positions - 2
        if max_residues_per_chunk <= 0:
            raise ValueError(f"invalid ESM2 max_position_embeddings: {max_positions}")

        pad_token_id = getattr(getattr(self.esm_model, "config", None), "pad_token_id", 0)
        chunk_ids = []
        chunk_spans = []
        for batch_idx in range(esm_input_ids.size(0)):
            valid_ids = esm_input_ids[batch_idx][esm_attention_mask[batch_idx].to(dtype=torch.bool)]
            if valid_ids.numel() < 3:
                continue
            prefix_id = valid_ids[:1]
            suffix_id = valid_ids[-1:]
            residue_ids = valid_ids[1:-1]
            start = 0
            while start < residue_ids.numel():
                end = min(start + max_residues_per_chunk, residue_ids.numel())
                if end == residue_ids.numel() and start > 0 and end - start < max_residues_per_chunk:
                    start = max(0, residue_ids.numel() - max_residues_per_chunk)
                    end = residue_ids.numel()
                chunk_residue_ids = residue_ids[start:end]
                chunk_ids.append(torch.cat((prefix_id, chunk_residue_ids, suffix_id), dim=0))
                chunk_spans.append((batch_idx, start, end - start))
                if end == residue_ids.numel():
                    break
                start = end

        if not chunk_ids:
            raise ValueError("ESM2 chunking did not produce any residue chunks")

        chunk_lengths = torch.tensor([ids.numel() for ids in chunk_ids], device=esm_input_ids.device)
        chunk_input_ids = torch.nn.utils.rnn.pad_sequence(
            chunk_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )
        chunk_positions = torch.arange(chunk_input_ids.size(1), device=esm_input_ids.device).unsqueeze(0)
        chunk_attention_mask = chunk_positions < chunk_lengths.unsqueeze(1)
        self.esm_model.eval()
        esm_outputs = self._run_esm_forward(
            input_ids=chunk_input_ids,
            attention_mask=chunk_attention_mask,
        )

        chunk_hidden = esm_outputs.last_hidden_state
        esm_hidden = chunk_hidden.new_zeros(
            esm_input_ids.size(0),
            max_residue_count,
            chunk_hidden.size(-1),
        )
        esm_hidden_counts = chunk_hidden.new_zeros(esm_input_ids.size(0), max_residue_count, 1)
        for chunk_idx, (batch_idx, start, length) in enumerate(chunk_spans):
            esm_hidden[batch_idx, start : start + length] += chunk_hidden[chunk_idx, 1 : 1 + length]
            esm_hidden_counts[batch_idx, start : start + length] += 1.0
        esm_hidden = esm_hidden / esm_hidden_counts.clamp_min(1.0)

        return esm_hidden.to(dtype=target_dtype), residue_mask

    def _compute_esm_hidden(
        self,
        esm_input_ids: torch.Tensor,
        esm_attention_mask: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.esm_model is None:
            raise ValueError("ESM2 is not initialized")
        residue_mask, max_residue_count = self._build_residue_mask_from_esm_attention(esm_attention_mask)
        max_positions = self._get_esm_max_positions()
        if max_positions is not None and esm_input_ids.size(1) > max_positions:
            return self._compute_chunked_esm_hidden(
                esm_input_ids=esm_input_ids,
                esm_attention_mask=esm_attention_mask,
                target_dtype=target_dtype,
                max_positions=max_positions,
                max_residue_count=max_residue_count,
                residue_mask=residue_mask,
            )

        self.esm_model.eval()
        esm_outputs = self._run_esm_forward(
            input_ids=esm_input_ids,
            attention_mask=esm_attention_mask,
        )
        esm_hidden = esm_outputs.last_hidden_state[:, 1 : 1 + max_residue_count]
        esm_hidden = esm_hidden.to(dtype=target_dtype)
        return esm_hidden, residue_mask

    def _run_esm_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Run the ESM2 forward pass with partial-unfreeze aware autograd.

        - Always keeps ``self.esm_model.eval()`` semantics (no dropout).
        - Drops the global ``torch.no_grad()`` so that unfrozen submodules can
          receive gradients.
        - Frozen submodules are detached via the standard ``requires_grad``
          path; autograd only flows through parameters that opt in.
        """
        self.esm_model.eval()
        return self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _cross_attention_fusion(
        self,
        esm_hidden: torch.Tensor,
        qwen_hidden: torch.Tensor,
        qwen_attention_mask: Optional[torch.Tensor],
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.esm_query_projection(esm_hidden)
        key = self.qwen_key_projection(qwen_hidden)
        value = self.qwen_value_projection(qwen_hidden)
        key_padding_mask = None
        if qwen_attention_mask is not None:
            key_padding_mask = ~qwen_attention_mask.to(device=qwen_hidden.device, dtype=torch.bool)
        fused_hidden = query * residue_mask.unsqueeze(-1).to(query.dtype)
        for fusion_layer in self.fusion_layers:
            fused_hidden = fusion_layer(
                residue_hidden=fused_hidden,
                qwen_key=key,
                qwen_value=value,
                key_padding_mask=key_padding_mask,
                residue_mask=residue_mask,
            )
        fused_hidden = self.fusion_norm(fused_hidden)
        return fused_hidden * residue_mask.unsqueeze(-1).to(fused_hidden.dtype)

    def compute_pooled_hidden(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        outputs = self.backbone.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs[0]
        if self.backbone_frozen:
            hidden_states = hidden_states.detach()
        return self._pool_last_token(hidden_states, attention_mask), outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        protein_labels=None,
        protein_label_mask=None,
        protein_token_mask=None,
        protein_sequence=None,
        esm_input_ids=None,
        esm_attention_mask=None,
        label=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        logits_to_keep=0,
        pooled_hidden=None,
        cache_index=None,
        threshold: float = 0.5,
        **kwargs,
    ):
        del labels, logits_to_keep, cache_index, protein_sequence
        kwargs.pop("output_assistant_text", None)
        if pooled_hidden is None:
            if not self.use_esm2 and protein_token_mask is None:
                raise ValueError("protein_token_mask is required for residue-level protein prediction")
            outputs = self.backbone.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
            hidden_states = outputs[0]
            if self.backbone_frozen:
                hidden_states = hidden_states.detach()
            if self.use_esm2:
                if esm_input_ids is None or esm_attention_mask is None:
                    raise ValueError("esm_input_ids and esm_attention_mask are required when use_esm2=True")
                head_param = next(self.protein_head.parameters())
                esm_hidden, residue_mask = self._compute_esm_hidden(
                    esm_input_ids=esm_input_ids.to(device=hidden_states.device),
                    esm_attention_mask=esm_attention_mask.to(device=hidden_states.device),
                    target_dtype=head_param.dtype,
                )
                protein_hidden = self._cross_attention_fusion(
                    esm_hidden=esm_hidden.to(device=head_param.device),
                    qwen_hidden=hidden_states.to(device=head_param.device, dtype=head_param.dtype),
                    qwen_attention_mask=attention_mask.to(device=head_param.device) if attention_mask is not None else None,
                    residue_mask=residue_mask.to(device=head_param.device),
                )
            else:
                protein_hidden, residue_mask = self._select_protein_hidden(hidden_states, protein_token_mask)
            backbone_hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
            backbone_attentions = outputs.attentions if hasattr(outputs, "attentions") else None
        else:
            backbone_hidden_states = None
            backbone_attentions = None
            head_param = next(self.protein_head.parameters())
            protein_hidden = pooled_hidden.to(device=head_param.device, dtype=head_param.dtype)
            if protein_hidden.ndim == 2:
                protein_hidden = protein_hidden.unsqueeze(1)
            residue_mask = torch.ones(
                protein_hidden.shape[:2],
                device=protein_hidden.device,
                dtype=torch.bool,
            )

        head_param = next(self.protein_head.parameters())
        protein_hidden = protein_hidden.to(device=head_param.device, dtype=head_param.dtype)
        residue_mask = residue_mask.to(device=head_param.device, dtype=torch.bool)
        logits = self.protein_head(protein_hidden)
        probabilities = torch.sigmoid(logits)
        pred_bits = probabilities >= threshold

        target = protein_labels if protein_labels is not None else label
        loss = None
        if target is not None:
            target = target.to(device=logits.device, dtype=torch.float32)
            if target.ndim == 3 and target.size(-1) == 1:
                target = target.squeeze(-1)
            logit_values = logits.squeeze(-1)
            if target.shape != logit_values.shape:
                raise ValueError(
                    f"protein label shape must match logits shape {tuple(logit_values.shape)}, "
                    f"got {tuple(target.shape)}"
                )
            if protein_label_mask is None:
                valid_mask = residue_mask
            else:
                valid_mask = protein_label_mask.to(device=logits.device, dtype=torch.bool)
                if valid_mask.shape != logit_values.shape:
                    raise ValueError(
                        f"protein label mask shape must match logits shape {tuple(logit_values.shape)}, "
                        f"got {tuple(valid_mask.shape)}"
                    )
                valid_mask = valid_mask & residue_mask
            loss = self._protein_residue_loss(logit_values, target, valid_mask)

        return S1ProteinOutput(
            loss=loss,
            logits=logits,
            probabilities=probabilities,
            pred_bits=pred_bits,
            backbone_hidden_states=backbone_hidden_states,
            backbone_attentions=backbone_attentions,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        state_dict = kwargs.pop("state_dict", None)
        esm_state_dict = None

        backbone_state_dict = {}
        decoder_state_dict = {}
        if state_dict is not None:
            normalized_state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }
            backbone_state_dict.update(
                {
                    key.removeprefix("backbone."): value
                    for key, value in normalized_state_dict.items()
                    if key.startswith("backbone.")
                }
            )
            decoder_state_dict.update(
                {
                    key: value
                    for key, value in normalized_state_dict.items()
                    if key.startswith(S1_PROTEIN_MODULE_PREFIXES)
                }
            )
            esm_state_dict = {
                key.removeprefix("esm_model."): value.detach().contiguous().cpu()
                for key, value in normalized_state_dict.items()
                if key.startswith("esm_model.") and torch.is_tensor(value)
            }
        else:
            backbone_state_dict.update(self.backbone.state_dict())

        for key, value in self.protein_head.state_dict().items():
            decoder_state_dict.setdefault(f"protein_head.{key}", value)
        if self.use_esm2:
            for module_name in (
                "esm_query_projection",
                "qwen_key_projection",
                "qwen_value_projection",
                "fusion_layers",
                "fusion_norm",
            ):
                module = getattr(self, module_name, None)
                if module is None:
                    continue
                for key, value in module.state_dict().items():
                    decoder_state_dict.setdefault(f"{module_name}.{key}", value)

        protein_config = self._protein_config_dict()
        vlm_dir = os.path.join(save_directory, S1_PROTEIN_VLM_DIR_NAME)
        self.backbone.save_pretrained(vlm_dir, state_dict=backbone_state_dict, **kwargs)
        self._save_protein_decoder(save_directory, decoder_state_dict, protein_config)
        self._save_esm2_pretrained(save_directory, state_dict=esm_state_dict)
        with open(os.path.join(save_directory, S1_PROTEIN_CONFIG_NAME), "w", encoding="utf-8") as f:
            json.dump(protein_config, f, ensure_ascii=False, indent=2)

    def _protein_config_dict(self) -> dict:
        return {
            "use_esm2": self.use_esm2,
            "esm_model_name": self.esm_model_name,
            "esm_fusion_dim": self.esm_fusion_dim,
            "esm_num_attention_heads": self.esm_num_attention_heads,
            "esm_fusion_num_layers": self.esm_fusion_num_layers,
            "esm_fusion_ffn_dim": self.esm_fusion_ffn_dim,
            "esm_unfreeze_last_n_layers": self.esm_unfreeze_last_n_layers,
            "esm_unfreeze_pooler": self.esm_unfreeze_pooler,
            "esm_unfreeze_final_layer_norm": self.esm_unfreeze_final_layer_norm,
            "esm_lr_multiplier": self.esm_lr_multiplier,
            "positive_loss_weight": self.positive_loss_weight,
            "protein_loss_type": self.protein_loss_type,
            "asl_gamma_pos": self.asl_gamma_pos,
            "asl_gamma_neg": self.asl_gamma_neg,
            "asl_clip": self.asl_clip,
            "asl_eps": self.asl_eps,
        }

    def _save_protein_decoder(
        self,
        save_directory: str,
        decoder_state_dict: dict[str, torch.Tensor],
        protein_config: dict,
    ) -> None:
        decoder_path = os.path.join(save_directory, S1_PROTEIN_DECODER_WEIGHTS_NAME)
        cpu_state_dict = {
            key: value.detach().contiguous().cpu() if torch.is_tensor(value) else value
            for key, value in decoder_state_dict.items()
        }
        torch.save(
            {
                "format": S1_PROTEIN_DECODER_FORMAT,
                "state_dict": cpu_state_dict,
                "config": protein_config,
            },
            decoder_path,
        )

    @classmethod
    def from_checkpoint(cls, model_name_or_path: str, **kwargs):
        return cls.from_pretrained(model_name_or_path, **kwargs)
