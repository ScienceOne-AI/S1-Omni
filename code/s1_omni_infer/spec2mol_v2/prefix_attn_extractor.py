"""Self-contained ``PrefixAttnExtractor`` for the spectra_omni motif predictor.

Copied from ``dataset/MAST_motif/spec2fg/scripts/train_prefix_attn.py`` so that
``spec2mol_v2`` no longer depends on the MAST_motif sweep tree being on sys.path.

Only the ``PrefixAttnExtractor`` class is needed. The original imports
``MotifHead`` from ``train_motif_predictor``, but that head is only used when
``feat_dim > 0``; the shipped extractor checkpoint uses ``feat_dim=0`` (head is a
plain ``nn.Sequential``), so the ``MotifHead`` dependency is dropped here. A
``feat_dim > 0`` build will raise a clear error rather than silently failing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PrefixAttnExtractor(nn.Module):
    """proj -> N x TransformerEncoderLayer -> Q-query cross-attn pooling -> head."""

    def __init__(
        self,
        in_dim: int = 5120,
        proj_dim: int = 768,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        num_queries: int = 4,
        num_classes: int = 20,
        feat_dim: int = 0,
        head_hidden: tuple[int, ...] = (512, 256),
        head_dropout: float = 0.1,
        frozen_head_state: dict | None = None,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if x_mean is None:
            x_mean = torch.zeros(in_dim)
        if x_std is None:
            x_std = torch.ones(in_dim)
        self.register_buffer("x_mean", x_mean.float())
        self.register_buffer("x_std", x_std.float().clamp_min(1e-6))

        self.input_ln = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, proj_dim)

        # learned positional bias added pre-encoder; sequence length is fixed at 546.
        self.pos = nn.Parameter(torch.zeros(1, 1, proj_dim))  # broadcast bias is enough
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.encoder_ln = nn.LayerNorm(proj_dim)

        # Q learned queries, cross-attention over encoded sequence.
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.empty(num_queries, proj_dim))
        nn.init.trunc_normal_(self.queries, std=proj_dim ** -0.5)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_ln = nn.LayerNorm(proj_dim)

        pooled_dim = num_queries * proj_dim
        self.feat_dim = int(feat_dim)
        self._head_frozen = False
        if self.feat_dim > 0:
            # Original code used a MotifHead here. The shipped checkpoint uses
            # feat_dim=0 so this branch is unused; raise to avoid a silent
            # mismatch if someone requests feat_dim>0.
            raise NotImplementedError(
                "PrefixAttnExtractor with feat_dim>0 needs MotifHead, which is "
                "not bundled in spec2mol_v2. The shipped extractor uses feat_dim=0."
            )
        else:
            self.to_feat = None
            self.head = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, head_hidden[0]),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden[0], num_classes),
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self._head_frozen and self.head is not None:
            self.head.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 5120]
        x = (x - self.x_mean) / self.x_std
        x = self.input_ln(x)
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        x = self.encoder_ln(x)
        return x

    def pool(self, encoded: torch.Tensor) -> torch.Tensor:
        B = encoded.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        out, _ = self.cross_attn(query=q, key=encoded, value=encoded, need_weights=False)
        out = self.cross_ln(out)  # [B, Q, D]
        return out.flatten(1)  # [B, Q*D]

    def forward(self, x: torch.Tensor):
        encoded = self.encode(x)
        pooled = self.pool(encoded)
        if self.to_feat is None:
            return self.head(pooled), None
        feat_pred = self.to_feat(pooled)
        logits = self.head(feat_pred)
        return logits, feat_pred
