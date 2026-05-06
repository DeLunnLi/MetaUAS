#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-image alignment (MetaUAS / gaobb/MetaUAS structure).
Fixes: list-prompt branch typo from upstream; hard-alignment uses softmax map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class AlignmentLayer(nn.Module):
    def __init__(self, input_channels: int = 2048, hidden_channels: int = 256, alignment_type: str = "sa"):
        super().__init__()
        self.alignment_type = alignment_type
        if alignment_type != "na":
            self.dimensionality_reduction = nn.Conv2d(
                input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, query_features: torch.Tensor, prompt_features: torch.Tensor) -> torch.Tensor:
        if self.alignment_type == "na":
            return prompt_features

        q = self.dimensionality_reduction(query_features)
        k = self.dimensionality_reduction(prompt_features)
        v = rearrange(prompt_features, "b c h w -> b c (h w)")

        soft_attention_map = torch.einsum("bcij,bckl->bijkl", q, k)
        soft_attention_map = rearrange(soft_attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
        soft_attention_map = torch.softmax(soft_attention_map, dim=-1)

        if self.alignment_type == "sa":
            return torch.einsum("bijp,bcp->bcij", soft_attention_map, v)

        if self.alignment_type == "ha":
            max_v, _ = soft_attention_map.max(dim=-1, keepdim=True)
            hard_attention_map = (soft_attention_map == max_v).float()
            return torch.einsum("bijp,bcp->bcij", hard_attention_map, v)

        raise ValueError(f"Unsupported alignment_type: {self.alignment_type}")


class AlignmentModule(nn.Module):
    def __init__(
        self,
        input_channels: int = 2048,
        hidden_channels: int = 256,
        alignment_type: str = "sa",
        fusion_policy: str = "cat",
    ):
        super().__init__()
        self.fusion_policy = fusion_policy
        self.alignment_layer = AlignmentLayer(input_channels, hidden_channels, alignment_type=alignment_type)

    def forward(self, query_features: torch.Tensor, prompt_features) -> torch.Tensor:
        if isinstance(prompt_features, list):
            aligned_list = [
                self.alignment_layer(query_features, prompt_features[i]) for i in range(len(prompt_features))
            ]
            aligned_prompt = torch.mean(torch.stack(aligned_list, dim=0), dim=0)
        else:
            aligned_prompt = self.alignment_layer(query_features, prompt_features)

        if self.fusion_policy == "cat":
            return rearrange([query_features, aligned_prompt], "two b c h w -> b (two c) h w")
        if self.fusion_policy == "add":
            return query_features + aligned_prompt
        if self.fusion_policy == "absdiff":
            return (query_features - aligned_prompt).abs()
        raise ValueError(f"Unsupported fusion_policy: {self.fusion_policy}")
