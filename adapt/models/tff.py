from __future__ import annotations

import torch
import torch.nn as nn


class TemporalFeatureFusion(nn.Module):
    def __init__(
        self,
        seq_len: int = 16,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head_fc1 = nn.Linear(embed_dim, 128)
        self.head_dropout = nn.Dropout(classifier_dropout)
        self.head_fc2 = nn.Linear(128, 1)
        self.act = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.norm(self.encoder(x))
        cls_repr = x[:, 0, :]
        hidden = self.head_dropout(self.act(self.head_fc1(cls_repr)))
        prob = torch.sigmoid(self.head_fc2(hidden)).squeeze(-1)
        aux = {"cls_repr": cls_repr, "token_sequence": x}
        return prob, aux

    def head_weight_matrices(self) -> list[torch.Tensor]:
        return [self.head_fc1.weight, self.head_fc2.weight]
