from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RewardTransformer(nn.Module):
    """Transformer-based reward model that fuses visual and optional language embeddings."""

    def __init__(
        self,
        args,
        video_dim: int = 768,
        text_dim: Optional[int] = None,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_stages: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.args = args

        # Multi-stage classification or continuous progress estimation
        self.discrete = args.discrete

        # Project video and text to common dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim is not None else None

        # Simple learnable positional embedding applied to the first token for stability
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.discrete:
            # Shared progress prediction head (applied to each frame)
            self.shared_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
            )

            # Stage classification head (applied to each frame)
            self.stage_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, num_stages),
                nn.Softmax(dim=-1),
            )

            # Progress estimation head (applied to each frame)
            self.progress_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.progress_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        max_tokens = args.max_seq_len + (1 if self.text_proj is not None else 0)
        base_mask = torch.triu(torch.ones(max_tokens, max_tokens) * float("-inf"), diagonal=1)
        self.register_buffer("attention_mask", base_mask, persistent=False)

    def _get_attention_mask(self, sequence_len: int, device: torch.device) -> torch.Tensor:
        mask = self.attention_mask[:sequence_len, :sequence_len]
        return mask.to(device)

    def forward(
        self,
        video_frames: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        video_embed[:, 0] += self.first_pos_embed

        text_embed_projected = None
        if text_embed is not None and self.text_proj is not None:
            text_embed_projected = self.text_proj(text_embed)

        if text_embed_projected is not None:
            sequence = torch.cat([text_embed_projected, video_embed], dim=1)
        else:
            sequence = video_embed

        mask = self._get_attention_mask(sequence.size(1), device=video_frames.device)
        transformed = self.transformer(sequence, mask=mask, is_causal=True)

        video_token_count = video_embed.shape[1]
        video_tokens = transformed[:, -video_token_count:, :]
        if self.discrete:
            stage_embedding = self.shared_head(video_tokens)
            stage_preds = self.stage_head(stage_embedding)
            progress_preds = self.progress_head(stage_embedding)
            return stage_preds, progress_preds
        else:
            progress_preds = self.progress_head(video_tokens)
            return None, progress_preds
