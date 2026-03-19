"""
KaniInferenceEngine: CUDA-graph-compatible inference engine for TTS.

This module provides a vLLM-style inference engine that:
1. Pre-computes all metadata (positions, frame tracking) using Python integers
2. Eliminates tensor value reads that break CUDA graph capture
3. Preserves BemaTTS frame-level position encoding
4. Uses StaticLfm2HybridConvCache for fixed-shape KV buffers (CUDA graph safe)

Architecture:
- Prefill phase: Variable length, uses standard HF forward with dynamic cache
- Decode phase: Fixed length (1 token), CUDA graph replay with static cache
"""

import gc
import torch
from typing import Optional
from kani_tts.context import KaniContext, set_context, reset_context
from kani_tts.model import compute_frame_level_positions
from kani_tts.static_cache import StaticLfm2HybridConvCache
from kani_tts.optimized_decode import OptimizedDecoder


class KaniInferenceEngine:
    """
    Metadata-based inference engine for KaniTTS.

    This engine replaces HuggingFace's generate() method with a custom loop that:
    - Computes all position IDs and metadata before forward pass (Python values only)
    - Uses StaticLfm2HybridConvCache with scatter_() for CUDA-graph-safe KV writes
    - Uses model.decode_step() which bypasses create_causal_mask()
    - Enables CUDA graph capture for decode phase

    Args:
        model: FlashCompatibleLfm2ForCausalLM instance
        audio_tokens_start: First audio token ID (typically 64410)
        tokens_per_frame: Number of audio tokens per frame (4 for NanoCodec)
        audio_step: Position step per frame (typically 1.0)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty factor
        use_cuda_graphs: Whether to use CUDA graphs for decode
    """

    def __init__(
        self,
        model,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        max_new_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        use_cuda_graphs: bool = False,
    ):
        self.model = model
        self.audio_tokens_start = audio_tokens_start
        self.tokens_per_frame = tokens_per_frame
        self.audio_step = audio_step
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.use_cuda_graphs = use_cuda_graphs

        # Ensure model is in eval mode (dropout in training mode gets baked into CUDA graphs)
        model.eval()

        # Disable SDPA math backend — it has .item() calls that break CUDA graphs.
        # Forces memory-efficient or flash attention instead.
        import torch.backends.cuda
        torch.backends.cuda.enable_math_sdp(False)

        self.device = next(model.parameters()).device
        self.vocab_size = model.config.vocab_size

        # Generation state
        self.audio_tokens_generated = 0
        self.current_frame_position = None

        # Audio logit mask: restrict sampling to audio tokens + EOS (lazy init in generate())
        # Only used when NOT using optimized decoder (fallback path)
        self._audio_logit_mask = None
        self._eos_token_id = None

        # Optimized decoder (lazy init on first generate() call when use_cuda_graphs=True)
        self.optimized_decoder = None

        # CUDA graph + static cache state (reused across generations)
        self.cuda_graph = None
        self.static_cache = None
        self.static_input_ids = None
        self.static_position_ids = None
        self.static_cache_position = None
        self.static_mask = None
        self.static_logits = None
        self._last_prefill_len = None

    def _prepare_prefill_metadata(
        self,
        input_ids: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> KaniContext:
        """
        Prepare metadata for prefill phase.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            speaker_emb: Optional speaker embedding [1, emb_dim]

        Returns:
            KaniContext with prefill metadata
        """
        batch_size, seq_len = input_ids.shape

        position_ids = compute_frame_level_positions(
            input_ids=input_ids,
            audio_tokens_start=self.audio_tokens_start,
            tokens_per_frame=self.tokens_per_frame,
            audio_step=self.audio_step,
        ).long()

        return KaniContext(
            is_prefill=True,
            num_prefill_tokens=seq_len,
            prefill_position_ids=position_ids,
            decode_position_ids=None,
            current_frame_position=None,
            audio_tokens_generated=0,
            past_seq_length=0,
            speaker_emb=speaker_emb,
        )

    def _prepare_decode_metadata(
        self,
        past_seq_length: int,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> KaniContext:
        """
        Prepare metadata for decode phase (single token generation).

        Args:
            past_seq_length: Number of tokens in KV-cache
            speaker_emb: Optional speaker embedding [1, emb_dim]

        Returns:
            KaniContext with decode metadata
        """
        if self.current_frame_position is None:
            self.current_frame_position = past_seq_length

        # Frame increment: check BEFORE this token's position is assigned
        # Tokens 0..3 → frame 0, tokens 4..7 → frame 1, etc.
        if self.audio_tokens_generated > 0 and self.audio_tokens_generated % self.tokens_per_frame == 0:
            self.current_frame_position += self.audio_step

        decode_position_ids = torch.tensor(
            [[self.current_frame_position]],
            dtype=torch.long,
            device=self.device
        )

        return KaniContext(
            is_prefill=False,
            num_prefill_tokens=0,
            prefill_position_ids=None,
            decode_position_ids=decode_position_ids,
            current_frame_position=self.current_frame_position,
            audio_tokens_generated=self.audio_tokens_generated,
            past_seq_length=past_seq_length,
            speaker_emb=speaker_emb,
        )

    def _finalize(self, generated_ids: torch.Tensor, gen_len: int) -> torch.Tensor:
        """Clone output slice, free the full buffer, and release cached CUDA memory."""
        result = generated_ids[:, :gen_len].clone()
        del generated_ids
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def _reset_generation_state(self):
        """Reset generation counters for a new sequence. Keeps CUDA graph alive for reuse."""
        self.audio_tokens_generated = 0
        self.current_frame_position = None
        reset_context()

    def _destroy_cuda_graph(self):
        """Fully release CUDA graph and all static tensors (for re-capture)."""
        self.cuda_graph = None
        self.static_cache = None
        self.static_mask = None
        self.static_input_ids = None
        self.static_position_ids = None
        self.static_cache_position = None
        self.static_logits = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _can_reuse_graph(self, prefill_len: int) -> bool:
        """Check if existing CUDA graph can be reused for a new generation."""
        if self.cuda_graph is None or self.static_cache is None:
            return False
        needed = prefill_len + self.max_new_tokens
        return needed <= self.static_cache.max_total_len

    def _refresh_static_cache(self, dynamic_cache, prefill_len: int):
        """
        Reset static cache DATA for a new generation, keeping tensors at same addresses.

        The CUDA graph references specific memory addresses (the static tensors).
        We keep the same tensor objects but overwrite their contents with fresh data
        from the new prefill + first decode. This avoids re-capturing the graph.
        """
        # 1. Zero out ALL KV data (ensures no stale data from previous generation)
        for layer_idx in range(len(self.static_cache.key_cache)):
            if self.static_cache.layer_types[layer_idx] == "full_attention":
                self.static_cache.key_cache[layer_idx].zero_()
                self.static_cache.value_cache[layer_idx].zero_()

        # 2. Copy fresh KV + conv data from new dynamic cache
        self.static_cache.copy_from_dynamic(dynamic_cache, prefill_len)

        # 3. Reset the mask: all -inf, then unmask prefill positions
        self.static_mask.fill_(float("-inf"))
        self.static_mask[:, :, :, :prefill_len] = 0.0

    def _initialize_cuda_graphs(self, dynamic_cache, prefill_len: int):
        """
        Initialize CUDA graphs with static KV cache.

        Args:
            dynamic_cache: Lfm2HybridConvCache from prefill + first decode
            prefill_len: Number of tokens processed so far (prefill + first decode)
        """
        # Over-allocate: assume worst-case prefill of 512 tokens so graphs can be reused
        max_prefill_for_alloc = max(prefill_len, 512)
        max_total_len = max_prefill_for_alloc + self.max_new_tokens
        config = self.model.config

        self._last_prefill_len = prefill_len
        print(f"   Initializing CUDA graphs: prefill_len={prefill_len}, max_total_len={max_total_len}")

        # 1. Create static cache with pre-allocated buffers
        self.static_cache = StaticLfm2HybridConvCache(
            config=config,
            max_total_len=max_total_len,
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # 2. Copy dynamic cache into static cache
        self.static_cache.copy_from_dynamic(dynamic_cache, prefill_len)

        # 3. Create static attention mask [1, 1, 1, max_total_len]
        self.static_mask = torch.full(
            (1, 1, 1, max_total_len),
            float("-inf"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.static_mask[:, :, :, :prefill_len] = 0.0

        # 4. Allocate static tensors for decode_step() inputs
        self.static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.static_position_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.static_cache_position = torch.zeros((1,), dtype=torch.long, device=self.device)

        # 5. Save conv cache before warmup (warmup + capture corrupt it)
        conv_cache_backup = [c.clone() for c in self.static_cache.conv_cache]

        # 6. Select decode function: optimized decoder or original decode_step
        decode_fn = self.optimized_decoder if self.optimized_decoder is not None else self.model.decode_step

        # 7. Warmup decode once before capture
        self.static_input_ids.fill_(1)
        self.static_position_ids.fill_(prefill_len)
        self.static_cache_position.fill_(prefill_len)
        self.static_cache.set_write_position(prefill_len)
        self.static_mask[:, :, :, prefill_len] = 0.0

        with torch.no_grad():
            warmup_logits = decode_fn(
                input_ids=self.static_input_ids,
                position_ids=self.static_position_ids,
                past_key_values=self.static_cache,
                causal_mask=self.static_mask,
                cache_position=self.static_cache_position,
            )
        self.static_logits = warmup_logits

        # Reset warmup position
        self.static_mask[:, :, :, prefill_len] = float("-inf")
        for layer_idx in range(len(self.static_cache.key_cache)):
            if self.static_cache.layer_types[layer_idx] == "full_attention":
                self.static_cache.key_cache[layer_idx][:, :, prefill_len, :] = 0
                self.static_cache.value_cache[layer_idx][:, :, prefill_len, :] = 0
        for i, backup in enumerate(conv_cache_backup):
            self.static_cache.conv_cache[i].copy_(backup)

        # 8. Capture CUDA graph
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.static_logits = decode_fn(
                input_ids=self.static_input_ids,
                position_ids=self.static_position_ids,
                past_key_values=self.static_cache,
                causal_mask=self.static_mask,
                cache_position=self.static_cache_position,
            )
        torch.cuda.synchronize()

        # Restore conv cache and reset KV at warmup position
        for i, backup in enumerate(conv_cache_backup):
            self.static_cache.conv_cache[i].copy_(backup)
        for layer_idx in range(len(self.static_cache.key_cache)):
            if self.static_cache.layer_types[layer_idx] == "full_attention":
                self.static_cache.key_cache[layer_idx][:, :, prefill_len, :] = 0
                self.static_cache.value_cache[layer_idx][:, :, prefill_len, :] = 0

        print("   CUDA graph captured successfully")

    def _execute_decode_graph(self, next_token_id: int, position_id: int, step_offset: int):
        """
        Execute CUDA graph for one decode step.

        Args:
            next_token_id: Token ID to decode
            position_id: Position ID for this token (BemaTTS frame position)
            step_offset: Absolute position in KV cache to write to

        Returns:
            logits tensor [1, 1, vocab_size]
        """
        self.static_input_ids[0, 0] = next_token_id
        self.static_position_ids[0, 0] = position_id
        self.static_cache_position[0] = step_offset
        self.static_cache.set_write_position(step_offset)
        self.static_mask[:, :, :, step_offset] = 0.0

        self.cuda_graph.replay()
        self.static_cache.advance_position()

        return self.static_logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample next token using topk-based top-p sampling.

        Handles two modes:
        - Reduced vocab (optimized decoder): logits are [1, num_reduced], maps back to full IDs
        - Full vocab (fallback): logits are [1, vocab_size], applies audio mask

        Args:
            logits: Model logits [batch_size, vocab_size_or_reduced]
            input_ids: Previously generated tokens [batch_size, seq_len] (full vocab IDs)

        Returns:
            next_token: Sampled token ID [batch_size, 1] (full vocab ID)
        """
        # Detect reduced vocab mode from logits shape (prefill logits are always full vocab)
        use_reduced = (
            self.optimized_decoder is not None
            and logits.shape[-1] != self.vocab_size
        )

        # Apply repetition penalty in-place
        if self.repetition_penalty != 1.0:
            unique_tokens = input_ids[0].unique()
            if use_reduced:
                # Map full vocab IDs to reduced indices
                reduced_idx = self.optimized_decoder.full_to_reduced[unique_tokens]
                valid = reduced_idx >= 0
                if valid.any():
                    valid_reduced = reduced_idx[valid]
                    penalty_logits = logits[0, valid_reduced]
                    logits[0, valid_reduced] = torch.where(
                        penalty_logits > 0,
                        penalty_logits / self.repetition_penalty,
                        penalty_logits * self.repetition_penalty,
                    )
            else:
                penalty_logits = logits[0, unique_tokens]
                logits[0, unique_tokens] = torch.where(
                    penalty_logits > 0,
                    penalty_logits / self.repetition_penalty,
                    penalty_logits * self.repetition_penalty,
                )

        # Apply audio mask (only needed for full vocab mode)
        if not use_reduced and self._audio_logit_mask is not None:
            logits.add_(self._audio_logit_mask)

        # Top-k extraction: get top 256 candidates
        k = min(256, logits.shape[-1])
        topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)

        # Apply temperature
        if self.temperature != 1.0:
            topk_logits = topk_logits / self.temperature

        # Apply top-p (nucleus) sampling on the small topk set
        if self.top_p < 1.0:
            probs = torch.softmax(topk_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            mask = torch.zeros_like(cumprobs, dtype=torch.bool)
            mask[..., 1:] = cumprobs[..., :-1] >= self.top_p
            topk_logits.masked_fill_(mask, float('-inf'))

        # Sample from filtered distribution
        probs = torch.softmax(topk_logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)

        # Map back to vocab indices
        reduced_sampled = topk_indices.gather(-1, idx)

        if use_reduced:
            # Map reduced index → full vocab ID
            next_token = self.optimized_decoder.reduced_to_full[reduced_sampled]
        else:
            next_token = reduced_sampled

        return next_token

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_token_id: int,
        speaker_emb: Optional[torch.Tensor] = None,
        token_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """
        Generate tokens using metadata-based inference with optional CUDA graphs.

        Flow:
        1. Handle speaker embedding: project and insert into inputs_embeds at position 1
        2. Prefill: standard forward_with_metadata() with HF dynamic cache
        3. First decode: standard forward_with_metadata() to establish cache
        4. Init CUDA graphs: copy dynamic cache -> static cache, capture graph
        5. Decode loop: _execute_decode_graph() for each subsequent token

        Audio token counting and frame positions:
        - audio_tokens_generated counts tokens AFTER they are fed to the model
        - Frame boundary: when count % tokens_per_frame == 0, advance frame
        - This ensures tokens 0-3 share frame P, tokens 4-7 share frame P+1, etc.

        Args:
            input_ids: Input token IDs [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            eos_token_id: EOS token ID to stop generation
            speaker_emb: Optional speaker embedding [1, emb_dim]
            token_callback: Optional callable(token_id: int) invoked for each generated token.
                When provided, uses synchronous EOS checking instead of pipelined EOS
                to avoid emitting garbage tokens.

        Returns:
            generated_ids: Full sequence [1, total_len]
        """
        assert input_ids.shape[0] == 1, "Only batch_size=1 supported"

        self._reset_generation_state()

        # ========== HANDLE SPEAKER EMBEDDING ==========
        # Insert speaker embedding into inputs_embeds at position 1
        # (mirrors prepare_inputs_for_generation in the HF generate path)
        inputs_embeds = None
        if speaker_emb is not None:
            projection_dtype = self.model.model.speaker_emb_projection.weight.dtype
            speaker_emb_cast = speaker_emb.to(self.device, dtype=projection_dtype)
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            speaker_emb_projected = self.model.model.speaker_emb_projection(speaker_emb_cast)
            speaker_emb_projected = speaker_emb_projected.unsqueeze(1)  # [1, 1, hidden_size]
            inputs_embeds = torch.cat([
                inputs_embeds[:, :1, :],        # First token embedding
                speaker_emb_projected,           # Speaker embedding at position 1
                inputs_embeds[:, 1:, :],         # Rest of token embeddings
            ], dim=1)
            # Extend attention mask for inserted speaker embedding
            attention_mask = torch.cat([
                attention_mask[:, :1],
                torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype),
                attention_mask[:, 1:],
            ], dim=1)

        # Track actual prefill length (includes speaker embedding if inserted)
        prefill_seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]

        # ========== PREFILL ==========
        if inputs_embeds is not None:
            # Speaker embedding inserted: all prefill tokens are text/special, sequential positions
            prefill_position_ids = torch.arange(prefill_seq_len, device=self.device).unsqueeze(0)
        else:
            # No speaker embedding: compute frame-level positions from input_ids
            prefill_position_ids = compute_frame_level_positions(
                input_ids=input_ids,
                audio_tokens_start=self.audio_tokens_start,
                tokens_per_frame=self.tokens_per_frame,
                audio_step=self.audio_step,
            ).long()

        prefill_context = KaniContext(
            is_prefill=True,
            num_prefill_tokens=prefill_seq_len,
            prefill_position_ids=prefill_position_ids,
            decode_position_ids=None,
            current_frame_position=None,
            audio_tokens_generated=0,
            past_seq_length=0,
            speaker_emb=None,  # Already inserted into inputs_embeds
        )
        set_context(**prefill_context.__dict__)

        with torch.no_grad():
            outputs = self.model.forward_with_metadata(
                input_ids=input_ids if inputs_embeds is None else None,
                context=prefill_context,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].clone()
        del outputs, inputs_embeds  # Free prefill activations + full logits tensor

        # ========== INITIALIZE OPTIMIZED DECODER (once) ==========
        if self.use_cuda_graphs and self.optimized_decoder is None:
            self.optimized_decoder = OptimizedDecoder(
                self.model, self.audio_tokens_start, eos_token_id
            )

        # ========== INITIALIZE AUDIO LOGIT MASK (once, fallback path only) ==========
        if self.optimized_decoder is None:
            if self._audio_logit_mask is None or self._eos_token_id != eos_token_id:
                self._eos_token_id = eos_token_id
                self._audio_logit_mask = torch.full(
                    (1, self.vocab_size), float('-inf'),
                    device=self.device, dtype=torch.bfloat16,
                )
                self._audio_logit_mask[0, self.audio_tokens_start:] = 0.0
                self._audio_logit_mask[0, eos_token_id] = 0.0

        # ========== PRE-ALLOCATE OUTPUT BUFFER ==========
        max_total = input_ids.shape[1] + self.max_new_tokens
        generated_ids = torch.zeros((1, max_total), dtype=torch.long, device=self.device)
        generated_ids[0, :input_ids.shape[1]] = input_ids[0]
        gen_len = input_ids.shape[1]

        next_token = self._sample_next_token(logits, generated_ids[:, :gen_len])
        next_token_id = next_token[0, 0].item()
        generated_ids[0, gen_len] = next_token_id
        gen_len += 1

        if token_callback is not None:
            token_callback(next_token_id)

        if next_token_id == eos_token_id:
            return self._finalize(generated_ids, gen_len)

        # ========== FIRST DECODE (without CUDA graphs, to establish cache) ==========
        decode_context = self._prepare_decode_metadata(prefill_seq_len)
        set_context(**decode_context.__dict__)

        with torch.no_grad():
            outputs = self.model.forward_with_metadata(
                input_ids=next_token,
                context=decode_context,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].clone()
        del outputs

        # Count token A (just fed). All decode tokens are audio until EOS.
        self.audio_tokens_generated += 1

        next_token = self._sample_next_token(logits, generated_ids[:, :gen_len])
        next_token_id = next_token[0, 0].item()
        generated_ids[0, gen_len] = next_token_id
        gen_len += 1

        if token_callback is not None:
            token_callback(next_token_id)

        if next_token_id == eos_token_id:
            del past_key_values
            return self._finalize(generated_ids, gen_len)

        # ========== INIT CUDA GRAPHS ==========
        prefill_len = prefill_seq_len + 1

        if self.use_cuda_graphs:
            if self._can_reuse_graph(prefill_len):
                self._refresh_static_cache(past_key_values, prefill_len)
            else:
                self._destroy_cuda_graph()
                self._initialize_cuda_graphs(past_key_values, prefill_len)
            del past_key_values

        # ========== PRE-COMPUTE POSITION SCHEDULE ==========
        # Frame positions are deterministic: prefill_pos + (audio_count // tokens_per_frame) * audio_step
        # At loop step s (starting from 2), audio_tokens_generated at entry = s - 1
        # Position = prefill_seq_len + ((s - 1) // tokens_per_frame) * audio_step
        # Indexing: i = s - 2, so audio_count = i + 1
        position_schedule = (
            prefill_seq_len
            + (torch.arange(self.max_new_tokens, device=self.device) + 1) // self.tokens_per_frame
            * self.audio_step
        ).long()

        # EOS token as a scalar tensor for GPU-side comparison
        eos_tensor = torch.tensor([eos_token_id], device=self.device, dtype=torch.long)

        # ========== DECODE LOOP ==========
        # Two modes:
        # 1. Pipelined EOS (no callback): launch GPU first, check previous EOS after.
        #    Overlaps CPU sync with GPU work. Cost: one wasted graph replay on EOS.
        # 2. Synchronous EOS (with callback): check EOS immediately after sampling.
        #    Required to avoid emitting garbage tokens to the callback.
        use_pipelined_eos = token_callback is None
        prev_is_eos = None  # Only used in pipelined mode

        for step in range(2, self.max_new_tokens):
            i = step - 2  # 0-based index into pre-computed schedule

            if self.use_cuda_graphs:
                # CUDA graph replay with pre-computed positions
                self.static_input_ids.copy_(next_token.view(1, 1))
                self.static_position_ids[0, 0] = position_schedule[i]
                step_offset = prefill_len + i
                self.static_cache_position[0] = step_offset
                self.static_cache.set_write_position(step_offset)
                self.static_mask[:, :, :, step_offset] = 0.0

                self.cuda_graph.replay()
                self.static_cache.advance_position()

                logits = self.static_logits[:, -1, :]
            else:
                position_id = position_schedule[i].item()
                decode_pos = torch.tensor([[position_id]], dtype=torch.long, device=self.device)
                decode_context = KaniContext(
                    is_prefill=False,
                    decode_position_ids=decode_pos,
                )
                set_context(**decode_context.__dict__)
                with torch.no_grad():
                    outputs = self.model.forward_with_metadata(
                        input_ids=next_token,
                        context=decode_context,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

            # 2. Sample next token
            next_token = self._sample_next_token(logits, generated_ids[:, :gen_len])
            next_token_id = next_token[0, 0].item()
            generated_ids[0, gen_len] = next_token_id
            gen_len += 1
            self.audio_tokens_generated += 1

            if use_pipelined_eos:
                # 3a. Pipelined: check EOS from PREVIOUS step (overlapped with GPU work)
                if prev_is_eos is not None and prev_is_eos.item():
                    # Previous token was EOS — this step's output is garbage, undo it
                    gen_len -= 1
                    self.audio_tokens_generated -= 1
                    break
                # Launch async EOS comparison for THIS step's token
                prev_is_eos = (next_token.view(-1) == eos_tensor)
            else:
                # 3b. Synchronous: invoke callback then check EOS immediately
                token_callback(next_token_id)
                if next_token_id == eos_token_id:
                    break

        else:
            # Loop completed without break — check the last EOS (pipelined mode only)
            if use_pipelined_eos and prev_is_eos is not None and prev_is_eos.item():
                gen_len -= 1
                self.audio_tokens_generated -= 1

        return self._finalize(generated_ids, gen_len)
