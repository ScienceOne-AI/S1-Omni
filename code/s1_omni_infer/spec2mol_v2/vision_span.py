"""Qwen3-VL vision-span helpers shared by spec2mol_v2 runtime modules."""
from __future__ import annotations

import torch

# Qwen3-VL special token ids used to slice one image span.
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655

# <|vision_start|> + 544 image_pad + <|vision_end|> for the 1024x552 reference image.
EXPECTED_VISION_SPAN = 546


def slice_vision_span(hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return ``[B, 546, D]`` vision-span hidden states for a single image."""
    batch = input_ids.size(0)
    starts: list[int] = []
    for i in range(batch):
        ids = input_ids[i]
        pos = (ids == VISION_START_ID).nonzero(as_tuple=False)
        if pos.numel() == 0:
            raise RuntimeError(f"sample {i}: no <|vision_start|> token in input_ids")
        start = int(pos[0].item())
        end_idx = start + EXPECTED_VISION_SPAN - 1
        if end_idx >= ids.size(0) or int(ids[end_idx].item()) != VISION_END_ID:
            got = int(ids[end_idx].item()) if end_idx < ids.size(0) else "OOB"
            raise RuntimeError(
                f"sample {i}: expected <|vision_end|> at pos {end_idx}, got token id {got}"
            )
        starts.append(start)
    spans = [
        hidden[i, starts[i] : starts[i] + EXPECTED_VISION_SPAN, :]
        for i in range(batch)
    ]
    return torch.stack(spans, dim=0)
