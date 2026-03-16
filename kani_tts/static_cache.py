"""
StaticLfm2HybridConvCache: Drop-in replacement for Lfm2HybridConvCache with
pre-allocated fixed-size KV buffers for CUDA graph compatibility.

The HF Lfm2HybridConvCache grows KV cache via torch.cat() each decode step,
creating tensors with different sizes and memory addresses. CUDA graphs require
all tensor shapes and addresses to be fixed across replays.

Solution: Pre-allocate max-size buffers and use scatter_() for in-place writes.
scatter_() takes tensor indices and runs entirely on GPU - no CPU reads.
"""

import torch
from typing import Optional
from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache


class StaticLfm2HybridConvCache:
    """
    Static KV cache for CUDA-graph-compatible LFM2 inference.

    Pre-allocates fixed-size KV buffers and uses scatter_() for writes.
    Conv cache is copied from the dynamic cache (already fixed-size).

    Usage:
        1. Run prefill with HF's dynamic Lfm2HybridConvCache
        2. Create StaticLfm2HybridConvCache with max_total_len
        3. Copy dynamic cache contents via copy_from_dynamic()
        4. Use this cache for all subsequent decode steps
    """

    def __init__(
        self,
        config,
        max_total_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        self.max_total_len = max_total_len
        self.batch_size = batch_size
        self._dtype = dtype
        self.device = device

        # Model dimensions
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        self.layer_types = list(config.layer_types) if hasattr(config, "layer_types") else ["full_attention"] * num_layers

        # Find first attention layer (needed for get_seq_length)
        self.first_attention_layer = 0
        for i, lt in enumerate(self.layer_types):
            if lt == "full_attention":
                self.first_attention_layer = i
                break

        # Pre-allocate KV buffers: [batch, kv_heads, max_total_len, head_dim]
        self.key_cache = []
        self.value_cache = []
        for layer_idx in range(num_layers):
            if self.layer_types[layer_idx] == "full_attention":
                k = torch.zeros(batch_size, num_kv_heads, max_total_len, head_dim, dtype=dtype, device=device)
                v = torch.zeros(batch_size, num_kv_heads, max_total_len, head_dim, dtype=dtype, device=device)
            else:
                # Conv layers don't use KV cache - store empty tensors
                k = torch.empty(0, dtype=dtype, device=device)
                v = torch.empty(0, dtype=dtype, device=device)
            self.key_cache.append(k)
            self.value_cache.append(v)

        # Conv cache: [batch, hidden_size, L_cache] - copy from dynamic cache
        conv_L_cache = getattr(config, "conv_L_cache", 4)
        self.conv_cache = [
            torch.zeros(batch_size, hidden_size, conv_L_cache, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # Write position tensor (GPU scalar) - used by scatter_() inside CUDA graph
        self._write_position = torch.zeros(1, dtype=torch.long, device=device)

        # Logical length: how many KV positions have valid data
        self._logical_length = 0

    def copy_from_dynamic(self, dynamic_cache: Lfm2HybridConvCache, logical_length: int):
        """
        One-time copy from HF's dynamic cache after prefill + first decode.

        Args:
            dynamic_cache: HF Lfm2HybridConvCache with variable-length KV data
            logical_length: Number of valid positions in the dynamic cache
        """
        self._logical_length = logical_length

        # Copy KV data into pre-allocated buffers
        for layer_idx in range(len(self.key_cache)):
            if self.layer_types[layer_idx] != "full_attention":
                continue
            if layer_idx < len(dynamic_cache.key_cache) and dynamic_cache.key_cache[layer_idx].numel() > 0:
                src_len = dynamic_cache.key_cache[layer_idx].shape[-2]
                self.key_cache[layer_idx][:, :, :src_len, :].copy_(dynamic_cache.key_cache[layer_idx])
                self.value_cache[layer_idx][:, :, :src_len, :].copy_(dynamic_cache.value_cache[layer_idx])

        # Copy conv cache (already fixed-size)
        for layer_idx in range(len(self.conv_cache)):
            if layer_idx < len(dynamic_cache.conv_cache):
                self.conv_cache[layer_idx].copy_(dynamic_cache.conv_cache[layer_idx])

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        """
        CUDA-graph-safe KV cache update using scatter_().

        Instead of torch.cat() (which changes tensor size), we write into a
        pre-allocated buffer at the current write position using scatter_().

        Args:
            key_states: [batch, kv_heads, 1, head_dim]
            value_states: [batch, kv_heads, 1, head_dim]
            layer_idx: Which layer to update

        Returns:
            (key_cache, value_cache) - full buffers (mask handles validity)
        """
        if key_states is not None and self.layer_types[layer_idx] == "full_attention":
            # Build scatter index: [batch, kv_heads, 1, head_dim]
            num_kv_heads = key_states.shape[1]
            head_dim = key_states.shape[3]
            idx = self._write_position.view(1, 1, 1, 1).expand(
                self.batch_size, num_kv_heads, 1, head_dim
            )
            # In-place write at current position
            self.key_cache[layer_idx].scatter_(2, idx, key_states)
            self.value_cache[layer_idx].scatter_(2, idx, value_states)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def set_write_position(self, pos: int):
        """Update write position tensor (call BEFORE graph replay)."""
        self._write_position.fill_(pos)

    def advance_position(self):
        """Increment logical length after a decode step."""
        self._logical_length += 1

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return logical length (number of valid KV positions)."""
        return self._logical_length

    def get_mask_sizes(self, cache_position, layer_idx):
        """Compatibility with HF interface."""
        return self._logical_length, 0

    def __len__(self):
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        """Zero out all cache tensors."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel() > 0:
                self.key_cache[layer_idx].zero_()
                self.value_cache[layer_idx].zero_()
        for layer_idx in range(len(self.conv_cache)):
            self.conv_cache[layer_idx].zero_()
        self._logical_length = 0
        self._write_position.fill_(0)
