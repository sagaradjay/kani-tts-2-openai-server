"""
OptimizedDecoder: Flat, fused decode forward pass for LFM2.

Replaces model.decode_step() with an optimized version that:
1. Fuses QKV projections (3→1 GEMV per attention layer)
2. Fuses gate+up projections (2→1 GEMV per MLP)
3. Reduces lm_head to audio-only tokens (165 MB → 33 MB per step)
4. Uses Triton fused kernels for RMSNorm, SiLU×mul, RoPE
5. Bypasses HuggingFace module overhead (method dispatch, kwargs, etc.)
"""

import torch
import torch.nn.functional as F
from kani_tts.triton_kernels import fused_rms_norm, fused_silu_mul, fused_rope


class OptimizedDecoder:
    """
    Optimized decode-only forward pass with fused weights.

    Extracts all weights from the HF model at init and reorganizes them
    for minimal kernel launches. The __call__ method is a flat loop
    with no Python class dispatch overhead.

    Args:
        model: FlashCompatibleLfm2ForCausalLM instance
        audio_tokens_start: First audio token ID (64410)
        eos_token_id: EOS token ID (7)
    """

    def __init__(self, model, audio_tokens_start: int, eos_token_id: int):
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16
        config = model.config

        # Model dimensions
        self.hidden_size = config.hidden_size  # 1024
        self.num_heads = config.num_attention_heads  # 16
        self.num_kv_heads = config.num_key_value_heads  # 8
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)  # 64
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 2
        self.scaling = self.head_dim ** -0.5
        self.q_dim = self.num_heads * self.head_dim  # 1024
        self.kv_dim = self.num_kv_heads * self.head_dim  # 512
        self.num_layers = config.num_hidden_layers  # 16
        self.layer_types = list(config.layer_types)
        self.eps = config.norm_eps  # 1e-5
        self.L_cache = config.conv_L_cache  # 3

        # Identify attention layer indices
        self.attn_layer_indices = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

        # ========== Extract and fuse weights ==========
        hf_model = model.model  # FlashCompatibleLfm2Model
        hf_layers = hf_model.layers

        # Embedding (shared with lm_head due to tie_word_embeddings)
        self.embed_tokens = hf_model.embed_tokens

        # Per-layer weights
        self.operator_norm_w = []  # [hidden_size] per layer
        self.ffn_norm_w = []      # [hidden_size] per layer
        self.gate_up_w = []       # [2*intermediate, hidden_size] per layer (fused w1+w3)
        self.down_w = []          # [hidden_size, intermediate] per layer

        # Attention-specific (only for attention layers, None for conv layers)
        self.qkv_w = []           # [q_dim+2*kv_dim, hidden_size] (fused QKV)
        self.q_norm_w = []        # [head_dim]
        self.k_norm_w = []        # [head_dim]
        self.o_w = []             # [hidden_size, q_dim]
        self.inv_freqs = []       # [head_dim/2] pre-computed with learnable alpha

        # Conv-specific (only for conv layers, None for attention layers)
        self.in_proj_w = []       # [3*hidden, hidden]
        self.conv_w = []          # [hidden, 1, L_cache] → reshaped to [hidden, L_cache]
        self.out_proj_w = []      # [hidden, hidden]

        for layer_idx in range(self.num_layers):
            layer = hf_layers[layer_idx]

            # Operator norm + FFN norm (all layers)
            self.operator_norm_w.append(layer.operator_norm.weight.data)
            self.ffn_norm_w.append(layer.ffn_norm.weight.data)

            # Fused gate+up: cat(w1.weight, w3.weight) along dim 0
            mlp = layer.feed_forward
            gate_up = torch.cat([mlp.w1.weight.data, mlp.w3.weight.data], dim=0)
            self.gate_up_w.append(gate_up)
            self.down_w.append(mlp.w2.weight.data)

            if self.layer_types[layer_idx] == "full_attention":
                attn = layer.self_attn
                # Fused QKV: cat(q_proj, k_proj, v_proj) along dim 0
                qkv = torch.cat([
                    attn.q_proj.weight.data,
                    attn.k_proj.weight.data,
                    attn.v_proj.weight.data,
                ], dim=0)
                self.qkv_w.append(qkv)
                self.q_norm_w.append(attn.q_layernorm.weight.data)
                self.k_norm_w.append(attn.k_layernorm.weight.data)
                self.o_w.append(attn.out_proj.weight.data)

                # Pre-compute inv_freq with learnable alpha
                if hf_model.learnable_rope_layers is not None and hf_model.learnable_rope_layers[layer_idx] is not None:
                    rope_module = hf_model.learnable_rope_layers[layer_idx]
                    with torch.no_grad():
                        inv_freq = (rope_module.inv_freq_base * rope_module.alpha).float()
                else:
                    inv_freq = hf_model.pos_emb.inv_freq.float()
                self.inv_freqs.append(inv_freq)

                # Placeholders for conv
                self.in_proj_w.append(None)
                self.conv_w.append(None)
                self.out_proj_w.append(None)
            else:
                conv = layer.conv
                self.in_proj_w.append(conv.in_proj.weight.data)
                # Conv weight: [hidden, 1, L_cache] → [hidden, L_cache]
                self.conv_w.append(conv.conv.weight.data[:, 0, :])
                self.out_proj_w.append(conv.out_proj.weight.data)

                # Placeholders for attn
                self.qkv_w.append(None)
                self.q_norm_w.append(None)
                self.k_norm_w.append(None)
                self.o_w.append(None)
                self.inv_freqs.append(None)

        # Final norm
        self.final_norm_w = hf_model.embedding_norm.weight.data

        # ========== Reduced LM head ==========
        full_lm_head_weight = model.lm_head.weight.data  # [vocab_size, hidden_size]
        audio_weight = full_lm_head_weight[audio_tokens_start:]  # [16128, 1024]
        eos_weight = full_lm_head_weight[eos_token_id:eos_token_id + 1]  # [1, 1024]

        # Reduced vocab: [EOS, audio_token_0, audio_token_1, ...]
        self.audio_lm_head_weight = torch.cat([eos_weight, audio_weight], dim=0)  # [16129, 1024]
        self.num_reduced = self.audio_lm_head_weight.shape[0]

        # Mapping: reduced_idx → full vocab ID
        self.reduced_to_full = torch.cat([
            torch.tensor([eos_token_id], device=self.device, dtype=torch.long),
            torch.arange(audio_tokens_start, config.vocab_size, device=self.device, dtype=torch.long),
        ])  # [16129]

        # Reverse mapping: full vocab ID → reduced_idx (-1 if not in reduced vocab)
        self.full_to_reduced = torch.full(
            (config.vocab_size,), -1, device=self.device, dtype=torch.long
        )
        self.full_to_reduced[eos_token_id] = 0
        self.full_to_reduced[audio_tokens_start:] = torch.arange(
            1, config.vocab_size - audio_tokens_start + 1, device=self.device, dtype=torch.long
        )

        # Attention layer counter for indexing into attn-specific weight lists
        self._attn_weight_idx = []
        attn_count = 0
        for layer_idx in range(self.num_layers):
            if self.layer_types[layer_idx] == "full_attention":
                self._attn_weight_idx.append(attn_count)
                attn_count += 1
            else:
                self._attn_weight_idx.append(-1)

        print(f"   OptimizedDecoder initialized:")
        print(f"   - Fused QKV: {len(self.attn_layer_indices)} attention layers (3→1 GEMV each)")
        print(f"   - Fused gate+up: {self.num_layers} layers (2→1 GEMV each)")
        print(f"   - Reduced lm_head: {config.vocab_size} → {self.num_reduced} tokens "
              f"({self.audio_lm_head_weight.numel() * 2 / 1e6:.1f} MB vs "
              f"{full_lm_head_weight.numel() * 2 / 1e6:.1f} MB)")

    def __call__(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values,
        causal_mask: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized decode forward pass. Drop-in replacement for model.decode_step().

        Args:
            input_ids: [1, 1] token to decode
            position_ids: [1, 1] frame-level position
            past_key_values: StaticLfm2HybridConvCache
            causal_mask: [1, 1, 1, max_total_len]
            cache_position: [1] cache write position

        Returns:
            logits: [1, 1, num_reduced] (reduced vocab logits)
        """
        # Token embedding
        hidden = self.embed_tokens(input_ids)  # [1, 1, 1024]

        # Track which attention weight index we're on
        attn_idx = 0

        for layer_idx in range(self.num_layers):
            is_attn = self.layer_types[layer_idx] == "full_attention"

            # ---- Operator norm ----
            normed = fused_rms_norm(hidden, self.operator_norm_w[layer_idx], self.eps)

            if is_attn:
                # ---- Fused QKV projection ----
                qkv = F.linear(normed, self.qkv_w[layer_idx])  # [1, 1, 2048]
                q_raw, k_raw, v_raw = qkv.split(
                    [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
                )

                # Reshape to per-head: [1, 1, heads, head_dim]
                q = q_raw.view(1, 1, self.num_heads, self.head_dim)
                k = k_raw.view(1, 1, self.num_kv_heads, self.head_dim)

                # Q/K LayerNorm (per-head RMSNorm, weight is [head_dim])
                q = fused_rms_norm(q, self.q_norm_w[layer_idx], self.eps)
                k = fused_rms_norm(k, self.k_norm_w[layer_idx], self.eps)

                # Reshape for attention: [1, heads, 1, head_dim]
                q = q.reshape(1, self.num_heads, 1, self.head_dim)
                k = k.reshape(1, self.num_kv_heads, 1, self.head_dim)
                v = v_raw.reshape(1, self.num_kv_heads, 1, self.head_dim)

                # Compute RoPE (learnable, pre-computed inv_freq per layer)
                inv_freq = self.inv_freqs[layer_idx]  # [32]
                pos = position_ids[0, 0].float()
                freqs = inv_freq * pos  # [32]
                emb = torch.cat([freqs, freqs])  # [64]
                cos_val = emb.cos().to(self.dtype)
                sin_val = emb.sin().to(self.dtype)

                # Apply RoPE
                q, k = fused_rope(q, k, cos_val, sin_val)

                # KV cache update
                k_full, v_full = past_key_values.update(k, v, layer_idx)

                # Scaled dot-product attention
                attn_out = F.scaled_dot_product_attention(
                    q, k_full, v_full,
                    attn_mask=causal_mask,
                    scale=self.scaling,
                    enable_gqa=True,
                )  # [1, num_heads, 1, head_dim]

                # Output projection
                attn_out = attn_out.reshape(1, 1, self.q_dim)  # [1, 1, 1024]
                hidden = hidden + F.linear(attn_out, self.o_w[layer_idx])

                attn_idx += 1
            else:
                # ---- Conv layer (slow path) ----
                # in_proj: [1, 1, 1024] → [1, 1, 3072]
                BCx = F.linear(normed, self.in_proj_w[layer_idx])
                BCx = BCx.transpose(-1, -2)  # [1, 3072, 1]
                B, C, x_conv = BCx.chunk(3, dim=-2)  # each [1, 1024, 1]
                Bx = B * x_conv  # [1, 1024, 1]

                # Conv cache: roll left, write new, dot product
                conv_state = past_key_values.conv_cache[layer_idx]  # [1, 1024, 3]
                conv_state = conv_state.roll(shifts=-1, dims=-1)
                cp = cache_position.clamp(0, self.L_cache - 1)
                conv_state[:, :, cp] = Bx.to(dtype=conv_state.dtype)
                past_key_values.conv_cache[layer_idx].copy_(conv_state)

                conv_out = torch.sum(
                    conv_state * self.conv_w[layer_idx], dim=-1
                )  # [1, 1024]
                conv_out = conv_out.unsqueeze(-1)  # [1, 1024, 1]

                y = C * conv_out  # [1, 1024, 1]
                y = y.transpose(-1, -2)  # [1, 1, 1024]
                hidden = hidden + F.linear(y, self.out_proj_w[layer_idx])

            # ---- MLP: fused gate+up → silu×mul → down ----
            normed = fused_rms_norm(hidden, self.ffn_norm_w[layer_idx], self.eps)
            gate_up = F.linear(normed, self.gate_up_w[layer_idx])  # [1, 1, 9216]
            mlp_out = fused_silu_mul(gate_up)  # [1, 1, 4608]
            hidden = hidden + F.linear(mlp_out, self.down_w[layer_idx])

        # ---- Final norm + reduced lm_head ----
        hidden = fused_rms_norm(hidden, self.final_norm_w, self.eps)
        logits = F.linear(hidden, self.audio_lm_head_weight)  # [1, 1, 16129]

        return logits
