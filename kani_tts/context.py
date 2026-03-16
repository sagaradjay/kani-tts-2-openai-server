"""
KaniContext: Metadata for CUDA-graph-compatible inference.

This module provides the KaniContext dataclass which stores all metadata needed
for the forward pass, eliminating tensor value reads that break CUDA graph capture.

Key innovation: All control flow decisions (prefill vs decode, position IDs, frame tracking)
use Python integers/booleans instead of reading tensor values.
"""

from dataclasses import dataclass
import torch


@dataclass
class KaniContext:
    """
    Metadata for a single forward pass through the model.

    This replaces tensor reads (e.g., cache_position[0]) with pre-computed Python values,
    making the forward pass compatible with CUDA graphs.

    Attributes:
        is_prefill: Whether this is a prefill pass (True) or decode pass (False)
        num_prefill_tokens: Number of tokens in prefill (0 for decode)
        prefill_position_ids: Position IDs for prefill phase [num_prefill_tokens]
        decode_position_ids: Position IDs for decode phase [1] (single token)
        current_frame_position: Position ID for current audio frame (BemaTTS)
        audio_tokens_generated: Counter for audio tokens generated so far
        past_seq_length: Length of KV-cache (number of tokens processed so far)
        speaker_emb: Speaker embedding tensor [1, emb_dim]
    """

    is_prefill: bool = False
    num_prefill_tokens: int = 0
    prefill_position_ids: torch.Tensor | None = None
    decode_position_ids: torch.Tensor | None = None
    current_frame_position: int | None = None
    audio_tokens_generated: int = 0
    past_seq_length: int = 0
    speaker_emb: torch.Tensor | None = None


# Global context for use during forward pass
_CONTEXT = KaniContext()


def get_context() -> KaniContext:
    """Get the current global context."""
    return _CONTEXT


def set_context(
    is_prefill: bool,
    num_prefill_tokens: int = 0,
    prefill_position_ids: torch.Tensor | None = None,
    decode_position_ids: torch.Tensor | None = None,
    current_frame_position: int | None = None,
    audio_tokens_generated: int = 0,
    past_seq_length: int = 0,
    speaker_emb: torch.Tensor | None = None,
):
    """
    Set the global context for the next forward pass.

    This should be called before each model.forward() to provide metadata.
    """
    global _CONTEXT
    _CONTEXT = KaniContext(
        is_prefill=is_prefill,
        num_prefill_tokens=num_prefill_tokens,
        prefill_position_ids=prefill_position_ids,
        decode_position_ids=decode_position_ids,
        current_frame_position=current_frame_position,
        audio_tokens_generated=audio_tokens_generated,
        past_seq_length=past_seq_length,
        speaker_emb=speaker_emb,
    )


def reset_context():
    """Reset the global context to default values."""
    global _CONTEXT
    _CONTEXT = KaniContext()
