"""SimpleFold 在 S1-Omni 下的 transformers 风格包装。

参考 modeling_s1_protein.py 维护风格：
- 裸 nn.Module 子类（不走 PreTrainedModel）；
- classmethod from_pretrained(hf_dir)，输入由 scripts/convert_simplefold_ckpt_to_hf.py
  转换出的 HF 目录（包含 config.json + model.safetensors）；
- ESM2 走 lazy init（torch.hub），不作为本 module 的子模块、不参与 state_dict；
- forward 输入一条氨基酸序列字符串，输出 SimpleFoldOutput；
- 第一阶段仅支持 plain FoldingDiT（foldingdit_700M.yaml 配方），text/dssp 留接口；
- 不依赖 PyTorch Lightning / Hydra / OmegaConf —— 用显式工厂函数从 config dict 实例化。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors import safe_open
except ImportError:
    safe_load_file = None
    safe_open = None

from s1_omni.protein_folding.model.flow import LinearPath
from s1_omni.protein_folding.model.torch.architecture import FoldingDiT
from s1_omni.protein_folding.model.torch.blocks import HomogenTrunk, DiTBlock
from s1_omni.protein_folding.model.torch.layers import (
    EfficientSelfAttentionLayer,
    TimestepEmbedder,
)
from s1_omni.protein_folding.model.torch.pos_embed import (
    AbsolutePositionEncoding,
    AxialRotaryPositionEncoding,
    FourierPositionEncoding,
)
from s1_omni.protein_folding.model.torch.text_embedding_architecture import (
    FoldingDiTwithLLM,
)
from s1_omni.protein_folding.model.torch.sampler import EMSampler
from s1_omni.protein_folding.processor.protein_processor import ProteinDataProcessor
from s1_omni.protein_folding.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from s1_omni.protein_folding.boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from s1_omni.protein_folding.utils.boltz_utils import process_structure, save_structure
from s1_omni.protein_folding.utils.datamodule_utils import process_one_inference_structure
from s1_omni.protein_folding.utils.esm_utils import _af2_to_esm, esm_registry
from s1_omni.protein_folding.utils.fasta_utils import (
    check_fasta_inputs,
    download_fasta_utilities,
    process_fastas,
)


# ---------------------------------------------------------------------------
# 工厂函数：把 yaml/config dict（带 hydra _target_ 标签）实例化为 nn.Module。
# 第一阶段只覆盖 plain foldingdit_*.yaml 用到的子模块集合。
# ---------------------------------------------------------------------------

_TARGET_REGISTRY = {
    "model.torch.architecture.FoldingDiT": FoldingDiT,
    "model.torch.text_embedding_architecture.FoldingDiTwithLLM": FoldingDiTwithLLM,
    "model.torch.blocks.HomogenTrunk": HomogenTrunk,
    "model.torch.blocks.DiTBlock": DiTBlock,
    "model.torch.layers.EfficientSelfAttentionLayer": EfficientSelfAttentionLayer,
    "model.torch.layers.TimestepEmbedder": TimestepEmbedder,
    "model.torch.pos_embed.AbsolutePositionEncoding": AbsolutePositionEncoding,
    "model.torch.pos_embed.AxialRotaryPositionEncoding": AxialRotaryPositionEncoding,
    "model.torch.pos_embed.FourierPositionEncoding": FourierPositionEncoding,
}

# 顶层 _target_ → 是否需要 text condition（pooled_text_hidden 必传）
_TEXT_CONDITIONED_TARGETS = {
    "model.torch.text_embedding_architecture.FoldingDiTwithLLM",
}


def _instantiate_from_config(node: Any):
    """递归遍历 dict / list，遇到含 _target_ 的 dict 就 import 该 class 并构造实例。
    支持 _partial_: True —— 返回一个不立即调用的 partial 工厂（HomogenTrunk
    里 block 是 partial，循环里二次调用构造每一层）。"""
    if isinstance(node, list):
        return [_instantiate_from_config(x) for x in node]
    if not isinstance(node, dict):
        return node
    if "_target_" not in node:
        return {k: _instantiate_from_config(v) for k, v in node.items()}

    target = node["_target_"]
    if target not in _TARGET_REGISTRY:
        raise KeyError(
            f"Unknown hydra _target_ {target!r}; add it to _TARGET_REGISTRY in "
            f"modeling_simplefold.py before using this config."
        )
    cls = _TARGET_REGISTRY[target]
    is_partial = bool(node.get("_partial_", False))
    kwargs = {
        k: _instantiate_from_config(v)
        for k, v in node.items()
        if not k.startswith("_")
    }
    if is_partial:
        from functools import partial as _partial

        return _partial(cls, **kwargs)
    return cls(**kwargs)


def _build_folding_model_from_config(config: dict[str, Any]) -> nn.Module:
    """从 yaml dict 实例化 FoldingDiT(plain) 或 FoldingDiTwithLLM(text-conditioned)。

    顶层 `_target_` 必须在 `_TARGET_REGISTRY` 里登记过；config 顶层就是该 class 的
    所有构造参数（嵌套子模块由 `_instantiate_from_config` 递归处理）。"""
    target = config.get("_target_")
    if target not in _TARGET_REGISTRY:
        raise ValueError(
            f"SimpleFold modeling: 未知顶层 _target_ {target!r}，请在 _TARGET_REGISTRY "
            f"中注册对应的 class。已知: {sorted(_TARGET_REGISTRY)}"
        )
    cleaned = {
        k: v
        for k, v in config.items()
        if not k.startswith("_") and k not in ("architecture", "variant")
    }
    cleaned.pop("_ckpt_source", None)
    return _instantiate_from_config({"_target_": target, **cleaned})


# 兼容旧名（stage 1 时叫 _build_folding_dit_from_config）
_build_folding_dit_from_config = _build_folding_model_from_config


# ---------------------------------------------------------------------------
# 输出 dataclass：保留必要 metadata 以便 save_pdb / save_cif 落盘。
# ---------------------------------------------------------------------------


@dataclass
class SimpleFoldOutput:
    coordinates: torch.Tensor          # (N_atom, 3)  unscaled angstrom
    pad_mask: torch.Tensor             # (N_atom,)    bool
    structure: Any                     # boltz_data_pipeline.types.Structure
    record: Any                        # Record 对象（含 id / chains 等元信息）
    record_id: str
    sequence: str


# ---------------------------------------------------------------------------
# SimpleFold 主模型
# ---------------------------------------------------------------------------


class SimpleFold(nn.Module):
    """SimpleFold 的 transformers 风格包装。

    支持两种顶层架构（由 HF 目录里的 ``config.json["_target_"]`` 决定）：

    - ``model.torch.architecture.FoldingDiT`` —— plain 配方，不接受文本条件；
    - ``model.torch.text_embedding_architecture.FoldingDiTwithLLM`` —— 接收
      来自 VLM 的 ``<prot_st>`` token hidden state 作为全局条件，经
      ``llm2time_linear`` + ``text_feat_proj`` + ``text_gate`` 注入到 ``c_emb``。

    第二阶段后 ``forward(...)`` 把 ``pooled_text_hidden`` 当作 text-conditioned
    模型的**必填**入参；plain 模型若收到非 None 值会立刻报错。
    """

    def __init__(self, folding_model: nn.Module, config: dict[str, Any]):
        super().__init__()
        self.model = folding_model  # 主 diffusion 模型（FoldingDiT or FoldingDiTwithLLM）
        self.config = config
        self._is_text_conditioned = (
            config.get("_target_") in _TEXT_CONDITIONED_TARGETS
        )
        self._llm_hidden_size = int(config.get("llm_hidden_size", 0))

        # 延迟初始化的外部依赖：ESM2 + 数据预处理 + sampler
        self._esm_model: Optional[nn.Module] = None
        self._esm_dict = None
        self._af2_to_esm: Optional[torch.Tensor] = None
        self._tokenizer: Optional[BoltzTokenizer] = None
        self._featurizer: Optional[BoltzFeaturizer] = None
        self._ccd_cache_dir: Optional[Path] = None

    @property
    def is_text_conditioned(self) -> bool:
        return self._is_text_conditioned

    @property
    def llm_hidden_size(self) -> int:
        return self._llm_hidden_size

    # ------------------------------------------------------------------
    # checkpoint / weight loading
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "SimpleFold":
        model_dir = Path(model_name_or_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"SimpleFold.from_pretrained expects a directory produced by "
                f"convert_simplefold_ckpt_to_hf.py; got {model_dir}"
            )

        fused_config_path = model_dir / "simplefold_config.json"
        config_path = fused_config_path if fused_config_path.is_file() else model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        folding_model = _build_folding_model_from_config(config)

        simplefold_index_path = model_dir / "simplefold.safetensors.index.json"
        if simplefold_index_path.is_file():
            state_dict = cls._load_indexed_safetensors(simplefold_index_path)
        else:
            index_path = model_dir / "model.safetensors.index.json"
            weight_path = model_dir / "model.safetensors"
            if weight_path.is_file():
                if safe_load_file is None:
                    raise ImportError("safetensors is required to load SimpleFold checkpoints")
                state_dict = safe_load_file(str(weight_path), device="cpu")
            elif index_path.is_file():
                state_dict = cls._load_indexed_safetensors(index_path)
            else:
                weight_path = model_dir / "pytorch_model.bin"
                if not weight_path.is_file():
                    raise FileNotFoundError(
                        f"Neither simplefold.safetensors.index.json, model.safetensors, "
                        f"model.safetensors.index.json nor pytorch_model.bin found under {model_dir}"
                    )
                state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=False)

        folding_model.load_state_dict(state_dict, strict=True)
        folding_model = folding_model.to(dtype=dtype, device=device)
        folding_model.eval()

        model = cls(folding_model=folding_model, config=config)
        return model

    @staticmethod
    def _load_indexed_safetensors(index_path: Path) -> dict[str, torch.Tensor]:
        if safe_open is None:
            raise ImportError("safetensors is required to load SimpleFold checkpoints")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        prefix = index.get("metadata", {}).get("simplefold_prefix", "")
        state_dict: dict[str, torch.Tensor] = {}
        for filename in sorted(set(weight_map.values())):
            path = index_path.parent / filename
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for key, mapped_filename in weight_map.items():
                    if mapped_filename != filename:
                        continue
                    if key not in available:
                        raise KeyError(f"{key!r} missing from {path}")
                    out_key = key.removeprefix(prefix) if prefix else key
                    state_dict[out_key] = handle.get_tensor(key)
        return state_dict

    # ------------------------------------------------------------------
    # lazy init helpers
    # ------------------------------------------------------------------
    def _ensure_esm(self):
        if self._esm_model is not None:
            return
        esm_name = self.config.get("esm_model", "esm2_3B")
        if esm_name not in esm_registry:
            raise KeyError(
                f"esm_model={esm_name!r} not in esm_registry; "
                f"available: {list(esm_registry.keys())}"
            )
        esm_model, esm_dict = esm_registry[esm_name]()
        esm_model.eval()
        target_device = self._module_device()
        esm_model = esm_model.to(target_device)
        af2_to_esm = _af2_to_esm(esm_dict).to(target_device)

        self._esm_model = esm_model
        self._esm_dict = esm_dict
        self._af2_to_esm = af2_to_esm

    def _ensure_processors(self):
        if self._tokenizer is None:
            self._tokenizer = BoltzTokenizer()
        if self._featurizer is None:
            self._featurizer = BoltzFeaturizer()

    def _module_device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    def _module_dtype(self) -> torch.dtype:
        for p in self.model.parameters():
            return p.dtype
        return torch.float32

    def set_ccd_cache_dir(self, cache_dir: str | Path):
        """指定一个目录用于放 boltz 数据管线需要的 ccd.pkl + boltz1_conf.ckpt。
        若目录不存在 / 缺文件，调用 `_prepare_inference_workspace` 时会自动下载。"""
        self._ccd_cache_dir = Path(cache_dir)

    # ------------------------------------------------------------------
    # text-conditioning helpers
    # ------------------------------------------------------------------
    def _prepare_text_embedding(
        self, pooled_text_hidden: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """校验 pooled_text_hidden 与 ckpt 类型匹配，并归一化形状到 (1, llm_hidden_size)。
        plain ckpt 收到非 None 直接报错；text-conditioned ckpt 必须提供。"""
        if not self._is_text_conditioned:
            if pooled_text_hidden is not None:
                raise ValueError(
                    "this checkpoint was loaded from a plain FoldingDiT config and does "
                    "not consume text conditioning; pass pooled_text_hidden=None."
                )
            return None

        if pooled_text_hidden is None:
            raise ValueError(
                "this checkpoint is text-conditioned (FoldingDiTwithLLM); "
                "pooled_text_hidden is required. Expected a tensor of shape "
                f"(1, {self._llm_hidden_size}) or ({self._llm_hidden_size},), e.g. the "
                "VLM <prot_st>-token hidden state."
            )

        if not isinstance(pooled_text_hidden, torch.Tensor):
            raise TypeError(
                f"pooled_text_hidden must be a torch.Tensor, got {type(pooled_text_hidden).__name__}"
            )

        te = pooled_text_hidden
        if te.dim() == 1:
            te = te.unsqueeze(0)
        elif te.dim() == 3 and te.shape[1] == 1:
            te = te.squeeze(1)
        elif te.dim() > 2:
            te = te.reshape(te.shape[0], -1)

        if self._llm_hidden_size and te.shape[-1] != self._llm_hidden_size:
            raise ValueError(
                f"pooled_text_hidden last dim mismatch: expected "
                f"{self._llm_hidden_size}, got {te.shape[-1]}"
            )
        return te

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        sequence: str,
        *,
        seed: int = 42,
        num_steps: int = 500,
        tau: float = 0.1,
        nsample_per_protein: int = 1,
        sample_id: str = "sample",
        workspace: Optional[str | Path] = None,
        pooled_text_hidden: Optional[torch.Tensor] = None,
    ) -> SimpleFoldOutput:
        """单条氨基酸序列推理。返回 SimpleFoldOutput。

        参数
        -----
        sequence : str
            纯氨基酸序列；多链请用 ':' 分隔（与原 SimpleFold 同语义）。
        seed, num_steps, tau, nsample_per_protein :
            与原 SimpleFold 推理对齐的采样控制参数。
        sample_id :
            放进生成 fasta / record id 中；影响输出的命名。
        workspace :
            存放 process_fastas 产物（structures/*.npz + records/*.json + ccd.pkl）
            的目录；不传则使用 tempfile 临时目录。
        pooled_text_hidden :
            text-conditioned ckpt 的 **必填** 入参；形状 ``(llm_hidden_size,)`` 或
            ``(1, llm_hidden_size)``。语义为从 VLM 抽取的 ``<prot_st>`` token
            最后一层 hidden（5120-d 对应 S1-VL-32B）。

            - text 模型缺该输入 → ``ValueError``
            - plain 模型传非 None → ``ValueError``
        """
        text_embedding = self._prepare_text_embedding(pooled_text_hidden)

        self._ensure_esm()
        self._ensure_processors()

        device = self._module_device()

        if workspace is None:
            import tempfile

            workspace = tempfile.mkdtemp(prefix="simplefold_infer_")
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        # ---------- 写一份临时 fasta，让 boltz schema 走完 npz/record 落盘流程 ----------
        fasta_dir = workspace / "single_fasta"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = fasta_dir / f"{sample_id}.fasta"
        fasta_path.write_text(f">{sample_id}\n{sequence}\n", encoding="utf-8")

        cache_dir = self._ccd_cache_dir if self._ccd_cache_dir is not None else workspace / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        download_fasta_utilities(cache_dir)  # 已存在不会重新下载

        data = check_fasta_inputs(fasta_path)
        if not data:
            raise ValueError(f"No fasta records parsed from {fasta_path}")
        # boltz schema 把 fasta -> structures/*.npz + records/*.json + manifest.json
        process_fastas(data=data, out_dir=workspace, ccd_path=cache_dir / "ccd.pkl")

        struct_files = sorted(list(workspace.glob("structures/*.npz")))
        if len(struct_files) != 1:
            raise RuntimeError(
                f"expected exactly 1 structure npz, got {len(struct_files)} in {workspace}"
            )
        struct_file = struct_files[0]
        record_file = workspace / "records" / f"{struct_file.stem}.json"

        # ---------- 与原推理路径对齐的随机种子设置 ----------
        # 对应 inference.py:infer_structures_subset 里的 pl.seed_everything(args.seed)
        # 这里手动复刻：torch + cuda + numpy + random + PYTHONHASHSEED
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ---------- 数据预处理：tokenize / featurize / ESM forward ----------
        processor = ProteinDataProcessor(
            device=device,
            scale=16.0,
            ref_scale=5.0,
            multiplicity=1,
            inference_multiplicity=nsample_per_protein,
            backend="torch",
        )

        prep = process_one_inference_structure(
            struct_file,
            record_file,
            self._tokenizer,
            self._featurizer,
            processor,
            self._esm_model,
            self._esm_dict,
            self._af2_to_esm,
            use_dssp=False,
            dssp_dir=None,
            fasta_path=fasta_path,
        )
        if prep is None:
            raise RuntimeError(f"process_one_inference_structure returned None for {struct_file}")
        batch, structure, record = prep

        # ---------- text-conditioned: 把 pooled <prot_st> hidden 注入 batch ----------
        if text_embedding is not None:
            # FoldingDiTwithLLM.forward 期望 batch["text_embedding"]; 形状
            # (B, llm_hidden_size)。我们这里 B==nsample_per_protein。
            te = text_embedding.to(device=device, dtype=torch.float32)
            if te.dim() == 1:
                te = te.unsqueeze(0)
            if te.shape[0] == 1 and nsample_per_protein > 1:
                te = te.expand(nsample_per_protein, -1).contiguous()
            batch["text_embedding"] = te

        # ---------- 反向采样 ----------
        flow = LinearPath()
        sampler = EMSampler(
            num_timesteps=num_steps,
            t_start=1e-4,
            tau=tau,
            log_timesteps=True,
            w_cutoff=0.99,
        )

        noise = torch.randn_like(batch["coords"]).to(device)
        out_dict = sampler.sample(self.model, flow, noise, batch)
        out_dict = processor.postprocess(out_dict, batch)
        sampled_coord = out_dict["denoised_coords"].detach()
        pad_mask = batch["atom_pad_mask"]

        return SimpleFoldOutput(
            coordinates=sampled_coord[0].cpu(),
            pad_mask=pad_mask[0].cpu(),
            structure=structure,
            record=record,
            record_id=record.id,
            sequence=sequence,
        )

    # ------------------------------------------------------------------
    # 输出辅助
    # ------------------------------------------------------------------
    def save_cif(self, output: SimpleFoldOutput, path: str | Path) -> Path:
        return self._save(output, path, output_format="mmcif")

    def save_pdb(self, output: SimpleFoldOutput, path: str | Path) -> Path:
        return self._save(output, path, output_format="pdb")

    def _save(self, output: SimpleFoldOutput, path: str | Path, output_format: str) -> Path:
        from copy import deepcopy

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        structure_save = process_structure(
            deepcopy(output.structure),
            output.coordinates,
            output.pad_mask,
            output.record,
            backend="torch",
        )
        # save_structure 期望 (structure, prediction_dir, outname, output_format=..., plddts=None)
        save_structure(
            structure_save,
            path.parent,
            path.stem,
            output_format=output_format,
            plddts=None,
        )
        # save_structure 写出的文件名固定为 outname + 默认扩展（mmcif → .cif / pdb → .pdb），
        # 如果用户指定的扩展不同，把它 rename 过去。
        produced = path.parent / f"{path.stem}.{'cif' if output_format == 'mmcif' else 'pdb'}"
        if path != produced and produced.is_file():
            if path.is_file():
                path.unlink()
            produced.rename(path)
        return path
