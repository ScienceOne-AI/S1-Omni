# ==================================================
# 功能说明：定义融合text_embedding条件信息的FoldingDiT网络前向逻辑与模块结构。
# 使用方法：由训练配置实例化后在forward中输入结构特征与时间步；text_embedding需为Tensor（如1x2048或Bx2048），输出去噪坐标预测。
# 依赖环境：torch、transformers及项目内model/utils模块；安装命令：pip install torch transformers
# 生成时间：2026-04-22 14:40:00
# ==================================================
#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math
import torch
from torch import nn
from torch.nn import functional as F

from s1_omni.protein_folding.model.torch.layers import FinalLayer, ConditionEmbedder
from s1_omni.protein_folding.utils.esm_utils import esm_model_dict
# from transformers import Qwen3Model, AutoTokenizer


class FoldingDiTwithLLM(nn.Module):
    def __init__(
        self,
        trunk,
        time_embedder,
        aminoacid_pos_embedder,
        pos_embedder,
        atom_encoder_transformer,
        atom_decoder_transformer,
        llm_hidden_size=2560,
        # llm2latent=2304,
        hidden_size=1152,
        num_heads=16,
        atom_num_heads=4,
        output_channels=3,
        atom_hidden_size_enc=256,
        atom_hidden_size_dec=256,
        atom_n_queries_enc=32,
        atom_n_keys_enc=128,
        atom_n_queries_dec=32,
        atom_n_keys_dec=128,
        esm_model="esm2_3B",
        esm_dropout_prob=0.0,
        use_atom_mask=False,
        use_length_condition=True,
        text_proj_mode="mlp_deep",
        text_gate_mode="vector",
        text_embedding_diag_interval=0,
    ):
        super().__init__()
        self.pos_embedder = pos_embedder
        pos_embed_channels = pos_embedder.embed_dim
        self.aminoacid_pos_embedder = aminoacid_pos_embedder
        aminoacid_pos_embed_channels = aminoacid_pos_embedder.embed_dim

        self.time_embedder = time_embedder

        self.atom_encoder_transformer = atom_encoder_transformer
        self.atom_decoder_transformer = atom_decoder_transformer

        # 不设 device_map，使 LLM 在 CPU 上初始化，由 Lightning/DDP 按 rank 移到各卡，避免整块 LLM 只落在 rank0
        self.llm_dtype = torch.bfloat16  # 显式指定，forward 中与下游 .float() 一致

        # 与 TimestepEmbedder 输出维一致（hidden_size）；TimestepEmbedder 无 .shape 属性
        self.llm2time_linear = nn.Linear(llm_hidden_size, hidden_size)
        # 中文注释：显式保存文本投影头模式，避免同名模块被多次覆盖
        self.text_proj_mode = str(text_proj_mode).lower()
        if self.text_proj_mode not in ("mlp_shallow", "linear_ln", "mlp_deep"):
            raise ValueError(
                f"Invalid text_proj_mode={text_proj_mode}. "
                f"Expected one of: ['mlp_shallow', 'linear_ln', 'mlp_deep']."
            )
        if self.text_proj_mode == "mlp_shallow":
            # 中文注释：单层投影 + 归一化 + 激活
            self.text_feat_proj = nn.Sequential(
                nn.Linear(llm_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
            )
        elif self.text_proj_mode == "linear_ln":
            # 中文注释：线性投影 + 归一化（无激活）
            self.text_feat_proj = nn.Sequential(
                nn.Linear(llm_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            # 中文注释：双层 MLP 投影，提供更强的非线性表达
            self.text_feat_proj = nn.Sequential(
                nn.Linear(llm_hidden_size, hidden_size),
                nn.SiLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            )
        self.text_gate_mode = str(text_gate_mode).lower()
        if self.text_gate_mode not in ("scalar", "vector", "none"):
            raise ValueError(
                f"Invalid text_gate_mode={text_gate_mode}. "
                f"Expected one of: ['scalar', 'vector', 'none']."
            )
        # 旧版：标量门控（保留用于消融）
        # self.text_gate = nn.Parameter(torch.full((1,), 1e-3))
        if self.text_gate_mode == "scalar":
            self.text_gate = nn.Parameter(torch.full((1,), 1e-3))
        elif self.text_gate_mode == "vector":
            # 新版：向量门控（按 hidden_size 逐通道缩放文本特征）
            self.text_gate = nn.Parameter(torch.full((hidden_size,), 1e-3))
        else:
            # 中文注释：none 模式不做门控，固定为 1 保持文本特征原样注入
            self.register_buffer("text_gate", torch.ones(1, dtype=torch.float32))
        print(
            f"[文本条件配置] text_proj_mode={self.text_proj_mode}, "
            f"text_gate_mode={self.text_gate_mode}"
        )
        self.text_embedding_scale_warn_threshold = 1e3
        self._text_embedding_diag_warn_count = 0
        self._text_embedding_diag_print_limit = 200
        self._text_embedding_diag_step_count = 0
        self.text_embedding_diag_interval = int(text_embedding_diag_interval)
        self.trunk = trunk

        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.atom_num_heads = atom_num_heads
        self.use_atom_mask = use_atom_mask
        self.esm_dropout_prob = esm_dropout_prob
        self.use_length_condition = use_length_condition

        esm_s_dim = esm_model_dict[esm_model]["esm_s_dim"]
        esm_num_layers = esm_model_dict[esm_model]["esm_num_layers"]

        self.atom_hidden_size_enc = atom_hidden_size_enc
        self.atom_hidden_size_dec = atom_hidden_size_dec
        self.atom_n_queries_enc = atom_n_queries_enc
        self.atom_n_keys_enc = atom_n_keys_enc
        self.atom_n_queries_dec = atom_n_queries_dec
        self.atom_n_keys_dec = atom_n_keys_dec

        atom_feat_dim = pos_embed_channels + aminoacid_pos_embed_channels + 427
        self.atom_feat_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.atom_pos_proj = nn.Linear(pos_embed_channels, hidden_size, bias=False)

        if self.use_length_condition:
            self.length_embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
            )

        self.atom_in_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        self.esm_s_combine = nn.Parameter(torch.zeros(esm_num_layers))
        self.esm_s_proj = ConditionEmbedder(
            input_dim=esm_s_dim,
            hidden_size=hidden_size,
            dropout_prob=self.esm_dropout_prob,
        )
        latent_cat_dim = hidden_size * 2
        self.esm_cat_proj = nn.Linear(latent_cat_dim, hidden_size)

        self.context2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        self.atom2latent_proj = nn.Sequential(
            nn.Linear(self.atom_hidden_size_enc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.atom_enc_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        self.atom_dec_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_dec),
            nn.LayerNorm(self.atom_hidden_size_dec),
        )

        self.latent2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.atom_hidden_size_dec),
        )

        self.final_layer = FinalLayer(
            self.atom_hidden_size_dec, 
            output_channels, 
            c_dim=hidden_size
        )
        #### 
        # self.encoder2llm_proj = nn.Sequential(
        #     nn.Linear(hidden_size, llm_hidden_size),
        #     nn.SiLU(),
        #     nn.LayerNorm(llm_hidden_size),
        # )
        # self.llm2trunk_proj = nn.Sequential(
        #     nn.Linear(llm_hidden_size, llm2latent),
        #     nn.SiLU(),
        #     nn.LayerNorm(llm2latent),
        # )
        ####

    def create_local_attn_bias(
        self, n: int, n_queries: int, n_keys: int, inf: float = 1e10, device: torch.device = None
    ) -> torch.Tensor:
        """Create local attention bias based on query window n_queries and kv window n_keys.

        Args:
            n (int): the length of quiries
            n_queries (int): window size of quiries
            n_keys (int): window size of keys/values
            inf (float, optional): the inf to mask attention. Defaults to 1e10.
            device (torch.device, optional): cuda|cpu|None. Defaults to None.

        Returns:
            torch.Tensor: the diagonal-like global attention bias
        """
        n_trunks = int(math.ceil(n / n_queries))
        padded_n = n_trunks * n_queries
        attn_mask = torch.zeros(padded_n, padded_n, device=device)
        for block_index in range(0, n_trunks):
            i = block_index * n_queries
            j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
            j2 = n_queries * block_index + (n_queries + n_keys) // 2
            attn_mask[i : i + n_queries, j1:j2] = 1.0
        attn_bias = (1 - attn_mask) * -inf
        return attn_bias.to(device=device)[:n, :n]

    def create_atom_attn_mask(
        self, 
        feats, 
        natoms, 
        atom_n_queries=None, 
        atom_n_keys=None,
        inf: float = 1e10
    ) -> torch.Tensor:
        if atom_n_queries is not None and atom_n_keys is not None:
            atom_attn_mask = self.create_local_attn_bias(
                n=natoms,
                n_queries=atom_n_queries,
                n_keys=atom_n_keys,
                device=feats["ref_pos"].device,
                inf=inf,
            )
        else:
            atom_attn_mask = None

        return atom_attn_mask

    def _extract_record_ids(self, feats):
        """从 batch 特征中提取可读的样本标识，便于回溯坏样本。"""
        record_info = feats.get("record", None)
        if record_info is None:
            return "未知样本"

        def _extract_single(item):
            if isinstance(item, dict):
                for key in ("id", "record_id", "name"):
                    value = item.get(key, None)
                    if isinstance(value, str) and value.strip():
                        return value
                return str(item)
            if isinstance(item, str):
                return item
            return str(item)

        if isinstance(record_info, (list, tuple)):
            if len(record_info) == 0:
                return "空样本列表"
            ids = [_extract_single(x) for x in record_info]
            return " | ".join(ids[:4]) + (" | ..." if len(ids) > 4 else "")
        return _extract_single(record_info)

    def _diagnose_text_embedding(self, feats, text_embedding: torch.Tensor):
        """在 text_feat_proj 前诊断 text_embedding，区分坏样本与尺度过大。"""
        if not isinstance(text_embedding, torch.Tensor) or text_embedding.numel() == 0:
            return
        with torch.no_grad():
            self._text_embedding_diag_step_count += 1
            emb_fp32 = text_embedding.detach().float()
            isfinite_mask = torch.isfinite(emb_fp32)
            isfinite_all = bool(isfinite_mask.all().item())
            finite_values = emb_fp32[isfinite_mask]
            finite_count = int(finite_values.numel())
            total_count = int(emb_fp32.numel())
            nonfinite_count = total_count - finite_count

            if finite_count > 0:
                abs_max = float(finite_values.abs().max().item())
                mean_val = float(finite_values.mean().item())
                var_val = float(finite_values.var(unbiased=False).item())
            else:
                abs_max = float("nan")
                mean_val = float("nan")
                var_val = float("nan")

            scale_too_large = abs_max > self.text_embedding_scale_warn_threshold
            has_bad_values = (not isfinite_all) or finite_count == 0
            need_periodic_print = (
                self.text_embedding_diag_interval > 0
                and self._text_embedding_diag_step_count % self.text_embedding_diag_interval == 0
            )
            if not has_bad_values and not scale_too_large and not need_periodic_print:
                return

            if (has_bad_values or scale_too_large):
                self._text_embedding_diag_warn_count += 1
            if self._text_embedding_diag_warn_count > self._text_embedding_diag_print_limit and not need_periodic_print:
                return

            record_ids = self._extract_record_ids(feats)
            if has_bad_values and scale_too_large:
                diagnosis = "疑似坏样本且尺度过大（同时存在NaN/Inf与超大数值）"
            elif has_bad_values:
                diagnosis = "疑似坏样本（检测到NaN/Inf）"
            else:
                if scale_too_large:
                    diagnosis = (
                        f"疑似尺度过大（|x|max={abs_max:.4e} > "
                        f"{self.text_embedding_scale_warn_threshold:.1e}）"
                    )
                else:
                    diagnosis = "周期统计（当前未发现明显异常）"

            print(
                "[文本嵌入诊断] "
                f"结论={diagnosis}; "
                f"isfinite={isfinite_all}; "
                f"abs_max={abs_max:.4e}; mean={mean_val:.4e}; var={var_val:.4e}; "
                f"finite={finite_count}/{total_count}; nonfinite={nonfinite_count}; "
                f"样本标识={record_ids}; "
                f"warn_count={self._text_embedding_diag_warn_count}; "
                f"step_count={self._text_embedding_diag_step_count}"
            )

    def forward(self, noised_pos, t, feats, self_cond=None):
        B, N, _ = feats["ref_pos"].shape
        M = feats["mol_type"].shape[1]
        atom_to_token = feats["atom_to_token"].float() # [B, N, M]
        ### [B, N]
        atom_to_token_idx = feats["atom_to_token_idx"] 
        ### [B, N]
        ref_space_uid = feats["ref_space_uid"]

        # create atom attention masks
        atom_attn_mask_enc = self.create_atom_attn_mask(
            feats, 
            natoms=N,
            atom_n_queries=self.atom_n_queries_enc,
            atom_n_keys=self.atom_n_keys_enc,
        )
        atom_attn_mask_dec = self.create_atom_attn_mask(
            feats,
            natoms=N,
            atom_n_queries=self.atom_n_queries_dec,
            atom_n_keys=self.atom_n_keys_dec,
        )

        # create condition embeddings for AdaLN
        c_emb = self.time_embedder(t)  # (B, D)
        # print('c_emb.shape: ', c_emb.shape)
        if self.use_length_condition:
            length = feats["max_num_tokens"].float().unsqueeze(-1)
            c_emb = c_emb + self.length_embedder(torch.log(length))
        # 将大模型的输入和t拼接
        text_embedding_batch = feats.get("text_embedding", None)  # [Qwen3-hidden_state]
        if isinstance(text_embedding_batch, (list, tuple)):
            text_embedding = text_embedding_batch[0]
        else:
            text_embedding = text_embedding_batch

        # text_embedding 是数值张量（例如 1x2048），不是字符串；按 Tensor 维度做对齐。
        if not isinstance(text_embedding, torch.Tensor) or text_embedding.numel() == 0:
            text_embedding = torch.zeros(
                (B, self.llm2time_linear.in_features),
                device=c_emb.device,
                dtype=c_emb.dtype,
            )
        else:
            text_embedding = text_embedding.to(c_emb.device)
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)
            elif text_embedding.dim() == 3 and text_embedding.shape[1] == 1:
                text_embedding = text_embedding.squeeze(1)
            elif text_embedding.dim() > 2:
                text_embedding = text_embedding.reshape(text_embedding.shape[0], -1)

            if text_embedding.shape[-1] != self.llm2time_linear.in_features:
                raise ValueError(
                    f"Invalid text_embedding hidden size: expected "
                    f"{self.llm2time_linear.in_features}, got {text_embedding.shape[-1]}"
                )

            if text_embedding.shape[0] == 1 and B > 1:
                text_embedding = text_embedding.repeat(B, 1)
            elif text_embedding.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch for text_embedding: expected {B}, got "
                    f"{text_embedding.shape[0]}"
                )

        self._diagnose_text_embedding(feats, text_embedding)
        c_emb_before_text = c_emb
        llm_hidden_state = self.text_feat_proj(text_embedding.float())  # (B, hidden_size)
        llm_hidden_state = self.text_gate * llm_hidden_state
        with torch.no_grad():
            # 记录文本分支注入强度，供 Lightning 侧写入 TensorBoard 观测。
            text_proj_norm = llm_hidden_state.detach().float().norm(dim=-1).mean()
            c_emb_norm = c_emb_before_text.detach().float().norm(dim=-1).mean()
            text_ratio = text_proj_norm / (c_emb_norm + 1e-8)
            # 向量门控在 TensorBoard 中记录均值，便于与旧版标量门控曲线对齐比较
            text_gate_stats = self.text_gate.detach().float()
            text_gate_value = text_gate_stats.mean()
            text_gate_min = text_gate_stats.min()
            text_gate_max = text_gate_stats.max()
        c_emb = torch.add(c_emb, llm_hidden_state)  # (B, D)
        # create atom features
        mol_type = feats["mol_type"]
        mol_type = F.one_hot(mol_type, num_classes=4).float()       # [B, M, 4]
        res_type = feats["res_type"].float()                        # [B, M, 33]
        pocket_feature = feats["pocket_feature"].float()            # [B, M, 4]
        res_feat = torch.cat(
            [mol_type, res_type, pocket_feature], 
        dim=-1)                                                     # [B, M, 41]
        atom_feat_from_res = torch.bmm(atom_to_token, res_feat)     # [B, N, 41]
        atom_res_pos = self.aminoacid_pos_embedder(
            pos=atom_to_token_idx.unsqueeze(-1).float()
        )
        ref_pos_emb = self.pos_embedder(pos=feats["ref_pos"])
        atom_feat = torch.cat(
            [
                ref_pos_emb,                                        # (B, N, PD1)
                atom_feat_from_res,                                 # (B, N, 41)
                atom_res_pos,                                       # (B, N, PD2)
                feats["ref_charge"].unsqueeze(-1),                  # (B, N, 1)
                feats["atom_pad_mask"].unsqueeze(-1),               # (B, N, 1)
                feats["ref_element"],                               # (B, N, 128)
                feats["ref_atom_name_chars"].reshape(B, N, 4 * 64), # (B, N, 256)
            ],
            dim=-1,
        )                                                           # (B, N, PD1+PD2+427)
        atom_feat = self.atom_feat_proj(atom_feat)                  # (B, N, D)

        atom_coord = self.pos_embedder(pos=noised_pos)              # (B, N, PD1)
        atom_coord = self.atom_pos_proj(atom_coord)                 # (B, N, D)
        ### (B, N, D+D)
        atom_in = torch.cat([atom_feat, atom_coord], dim=-1)        
        atom_in = self.atom_in_proj(atom_in)                        # (B, N, D)

        # position embeddings for Axial RoPE
        atom_pe_pos = torch.cat(
            [
                ref_space_uid.unsqueeze(-1).float(),                 # (B, N, 1)
                feats["ref_pos"],                                    # (B, N, 3)
            ],
            dim=-1,
        )                                                            # (B, N, 4)
        token_pe_pos = torch.cat(
            [
                feats["residue_index"].unsqueeze(-1).float(),        # (B, M, 1)
                feats["entity_id"].unsqueeze(-1).float(),            # (B, M, 1)
                feats["asym_id"].unsqueeze(-1).float(),              # (B, M, 1)
                feats["sym_id"].unsqueeze(-1).float(),               # (B, M, 1)
            ],
            dim=-1,
        )                                                            # (B, M, 4)

        # atom encoder
        atom_c_emb_enc = self.atom_enc_cond_proj(c_emb)
        atom_latent = self.context2atom_proj(atom_in)
        atom_latent = self.atom_encoder_transformer(
            latents=atom_latent, 
            c=atom_c_emb_enc, 
            attention_mask=atom_attn_mask_enc,
            pos=atom_pe_pos,
        )
        atom_latent = self.atom2latent_proj(atom_latent)

        # grouping: aggregate atom tokens to residue tokens
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        latent = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_latent)
        assert latent.shape[1] == M

        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ feats['esm_s']).squeeze(2)
        force_drop_ids = feats.get("force_drop_ids", None)
        esm_emb = self.esm_s_proj(esm_s, self.training, force_drop_ids)

        if esm_emb.shape[0] != latent.shape[0]:
            if latent.shape[0] % esm_emb.shape[0] != 0:
                raise RuntimeError(
                    f"ESM batch mismatch cannot be aligned: latent={tuple(latent.shape)} esm_emb={tuple(esm_emb.shape)}"
                )
            repeat_factor = latent.shape[0] // esm_emb.shape[0]
            esm_emb = esm_emb.repeat_interleave(repeat_factor, dim=0)

        if esm_emb.dim() == 3 and esm_emb.shape[0] == latent.shape[1] and esm_emb.shape[1] == latent.shape[0]:
            esm_emb = esm_emb.permute(1, 0, 2).contiguous()

        if esm_emb.shape[1] != latent.shape[1]:
            raise RuntimeError(
                f"ESM token mismatch after alignment: latent={tuple(latent.shape)} esm_emb={tuple(esm_emb.shape)}"
            )

        if esm_emb.shape[:2] != latent.shape[:2]:
            raise RuntimeError(
                f"ESM/latent leading dims mismatch before cat: latent={tuple(latent.shape)} esm_emb={tuple(esm_emb.shape)}"
            )

        print(f"[TEXT_SHAPE_DEBUG] latent={tuple(latent.shape)} esm_emb={tuple(esm_emb.shape)}")
        latent = self.esm_cat_proj(torch.cat([latent, esm_emb], dim=-1))

        # residue trunk
        latent = self.trunk(
            latents=latent, 
            c=c_emb, 
            attention_mask=None,
            pos=token_pe_pos,
        )

        # ungrouping: broadcast residue tokens to atom tokens
        output = torch.bmm(atom_to_token, latent)
        assert output.shape[1] == N

        # add skip connection
        output = output + atom_latent
        output = self.latent2atom_proj(output)

        # atom decoder
        atom_c_emb_dec = self.atom_dec_cond_proj(c_emb)
        output = self.atom_decoder_transformer(
            latents=output, 
            c=atom_c_emb_dec,
            attention_mask=atom_attn_mask_dec,
            pos=atom_pe_pos,
        )
        output = self.final_layer(output, c=c_emb)

        return {
            "predict_velocity": output,
            "latent": latent,
            "text_debug_metrics": {
                "text_proj_norm": text_proj_norm,
                "c_emb_norm": c_emb_norm,
                "text_ratio": text_ratio,
                "text_gate": text_gate_value,
                "text_gate_min": text_gate_min,
                "text_gate_max": text_gate_max,
            },
        }
