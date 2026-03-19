"""Core components for Kani-TTS audio generation."""
import torch
from nemo.collections.tts.models import AudioCodecModel
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import os
import time

# Import BemaTTS custom model
from .model import FlashCompatibleLfm2ForCausalLM
from .inference_engine import KaniInferenceEngine


@dataclass
class TTSConfig:
    """Configuration for TTS model."""
    device_map: str = "auto"
    tokeniser_length: int = 64400
    start_of_text: int = 1
    end_of_text: int = 2
    max_new_tokens: int = 1200
    temperature: float = 1
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    nanocodec_model:str = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
    sample_rate = 22050

    # BemaTTS configuration
    text_vocab_size: int = 64400  # Text vocabulary size (for position encoding)
    tokens_per_frame: int = 4  # Number of audio tokens per frame
    audio_step: float = 0.5  # Position step size per audio frame

    # Learnable RoPE configuration
    use_learnable_rope: bool = False  # Enable learnable RoPE with per-layer alpha
    alpha_min: float = 0.1  # Minimum value for alpha (frequency scaling)
    alpha_max: float = 2.0  # Maximum value for alpha (frequency scaling)

    # Speaker embedding configuration
    speaker_emb_dim: int = 192  # Dimension of speaker embeddings

    # Attention implementation
    attn_implementation: str = "sdpa"  # "sdpa", "flash_attention_2", or "eager"

    # CUDA graphs optimization
    use_cuda_graphs: bool = False  # Enable CUDA graphs for decode phase (1.3-1.5x speedup)


class NemoAudioPlayer:
    """Handles audio codec operations using NVIDIA NeMo."""

    def __init__(self, config: TTSConfig, text_tokenizer_name: Optional[str] = None) -> None:
        self.conf = config
        self.nemo_codec_model = AudioCodecModel\
                .from_pretrained(self.conf.nanocodec_model).eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nemo_codec_model.to(self.device)
        self.text_tokenizer_name = text_tokenizer_name
        if self.text_tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_name)

        self.tokeniser_length = self.conf.tokeniser_length
        self.start_of_text = self.conf.start_of_text
        self.end_of_text = self.conf.end_of_text
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

    def output_validation(self, out_ids: torch.Tensor) -> None:
        """Validate that output contains required speech markers."""
        start_of_speech_flag = self.start_of_speech in out_ids
        end_of_speech_flag = self.end_of_speech in out_ids
        if not (start_of_speech_flag and end_of_speech_flag):
            raise ValueError('Special speech tokens not exist!')

    def get_nano_codes(self, out_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and process audio codes from model output."""
        start_a_idx = (out_ids == self.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_a_idx   = (out_ids == self.end_of_speech).nonzero(as_tuple=True)[0].item()
        if start_a_idx >= end_a_idx:
            raise ValueError('Invalid audio codes sequence!')

        audio_codes = out_ids[start_a_idx+1 : end_a_idx]
        # Truncate to nearest complete frame (model may emit end_of_speech mid-frame)
        remainder = len(audio_codes) % 4
        if remainder:
            audio_codes = audio_codes[:len(audio_codes) - remainder]
        if len(audio_codes) == 0:
            raise ValueError('No complete audio frames generated!')
        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes - torch.tensor([self.codebook_size * i for i in range(4)])
        audio_codes = audio_codes - self.audio_tokens_start
        if (audio_codes < 0).sum().item() > 0:
            raise ValueError('Invalid audio tokens!')

        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]])
        return audio_codes, len_

    def get_text(self, out_ids: torch.Tensor) -> str:
        """Extract text from token sequence."""
        start_t_idx = (out_ids == self.start_of_text).nonzero(as_tuple=True)[0].item()
        end_t_idx   = (out_ids == self.end_of_text).nonzero(as_tuple=True)[0].item()
        txt_tokens = out_ids[start_t_idx : end_t_idx+1]
        text = self.tokenizer.decode(txt_tokens, skip_special_tokens=True)
        return text

    def get_waveform(self, out_ids: torch.Tensor) -> Tuple[np.ndarray, Optional[str]]:
        """Convert model output tokens to audio waveform."""
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)
        audio_codes, len_ = audio_codes.to(self.device), len_.to(self.device)
        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=audio_codes, tokens_len=len_)
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        if self.text_tokenizer_name:
            text = self.get_text(out_ids)
            return output_audio, text
        else:
            return output_audio, None

    def decode_audio_chunk(self, audio_codes) -> Optional[np.ndarray]:
        """Decode a chunk of audio codes for streaming.

        Args:
            audio_codes: numpy array of shape [num_frames, 4] with raw token IDs

        Returns:
            Audio waveform as numpy array, or None if invalid tokens
        """
        if len(audio_codes) == 0:
            return None

        audio_codes = torch.tensor(audio_codes, device=self.device)
        audio_codes = audio_codes - torch.tensor(
            [self.codebook_size * i for i in range(4)], device=self.device
        )
        audio_codes = audio_codes - self.audio_tokens_start

        if (audio_codes < 0).sum().item() > 0:
            return None  # Invalid tokens, skip

        # Shape: (1, 4, num_frames)
        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]], device=self.device)

        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(
                tokens=audio_codes, tokens_len=len_
            )
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        return output_audio


class KaniModel:
    """Text-to-speech model using causal language model."""

    def __init__(self, config: TTSConfig, model_name: str, player: NemoAudioPlayer) -> None:
        self.conf = config
        self.player = player
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.conf.use_learnable_rope:
            print("🚀 Loading model with BemaTTS + Learnable RoPE...")
        else:
            print("🚀 Loading model with BemaTTS frame-level position encoding...")
        self.model = FlashCompatibleLfm2ForCausalLM.from_pretrained(
            model_name,
            audio_tokens_start=self.player.audio_tokens_start,
            tokens_per_frame=self.conf.tokens_per_frame,
            audio_step=self.conf.audio_step,
            use_learnable_rope=self.conf.use_learnable_rope,
            alpha_min=self.conf.alpha_min,
            alpha_max=self.conf.alpha_max,
            speaker_emb_dim=self.conf.speaker_emb_dim,
            torch_dtype=torch.bfloat16,
            device_map=self.conf.device_map,
            attn_implementation=self.conf.attn_implementation,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.speaker_settings = getattr(self.model.config, 'speaker_settings', None)
        self.status = 'singlspeaker'
        self.speaker_list = []
        if self.speaker_settings is not None:
            self.status = self.speaker_settings.get('status')
            self.speaker_list = self.speaker_settings.get('speaker_list', [])

        # Initialize inference engine
        self.inference_engine = None
        if torch.cuda.is_available():
            use_graphs = self.conf.use_cuda_graphs
            print(f"🚀 Initializing inference engine (CUDA graphs={'on' if use_graphs else 'off'})...")
            if use_graphs:
                self._patch_lfm2_for_cuda_graphs()
            self.inference_engine = KaniInferenceEngine(
                model=self.model,
                audio_tokens_start=self.player.audio_tokens_start,
                tokens_per_frame=self.conf.tokens_per_frame,
                audio_step=self.conf.audio_step,
                max_new_tokens=self.conf.max_new_tokens,
                temperature=self.conf.temperature,
                top_p=self.conf.top_p,
                repetition_penalty=self.conf.repetition_penalty,
                use_cuda_graphs=use_graphs,
            )
            print("✅ Inference engine ready")

        # Old CUDA graph variables (deprecated, kept for reference)
        self.cuda_graph = None
        self.static_input_ids = None
        self.static_position_ids = None
        self.static_cache_position = None
        self.static_outputs = None
        self.static_past_key_values = None

    def get_input_ids(self, text_prompt: str, speaker_id: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tokens with special markers."""
        if speaker_id is not None:
            text_prompt = f"{speaker_id.strip()}: {text_prompt}"

        START_OF_HUMAN = self.player.start_of_human
        END_OF_TEXT = self.player.end_of_text
        END_OF_HUMAN = self.player.end_of_human
        START_OF_AI = self.player.start_of_ai
        START_OF_SPEECH = self.player.start_of_speech

        input_ids = self.tokenizer(text_prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        # Include START_OF_AI and START_OF_SPEECH in prefill
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
        return modified_input_ids, attention_mask

    def _patch_lfm2_for_cuda_graphs(self):
        """
        Monkey-patch LFM2's ShortConv layer to avoid cache_position[0] reads.

        The HuggingFace implementation has:
            if past_key_values is not None and cache_position[0] > 0:

        This reads a tensor value to CPU, which is incompatible with CUDA graphs.

        Solution (vLLM-style): Use shape-based checks instead of value-based checks.
        - Single token decode: cache_position.shape[0] == 1
        - This doesn't require reading the tensor to CPU
        """
        try:
            from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv
        except ImportError:
            print("⚠️  Could not import Lfm2ShortConv - CUDA graphs may fail")
            return

        # Save original method
        original_slow_forward = Lfm2ShortConv.slow_forward

        def patched_slow_forward(
            self,
            x: torch.Tensor,
            past_key_values=None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            """
            Patched version of slow_forward that avoids reading cache_position[0].

            ✅ CUDA-graph compatible: Use shape checks instead of value checks
            """
            from transformers.models.lfm2.modeling_lfm2 import apply_mask_to_padding_states

            seqlen = x.shape[1]
            x = apply_mask_to_padding_states(x, attention_mask)
            BCx = self.in_proj(x).transpose(-1, -2)
            B, C, x = BCx.chunk(3, dim=-2)
            Bx = B * x

            # ✅ CUDA-graph compatible: check cache_position.numel() instead of cache_position[0]
            # numel() == 1 means single-token decode, numel() > 1 means prefill
            if past_key_values is not None and cache_position is not None and cache_position.numel() == 1:
                # Decode phase: update existing conv state (single token)
                conv_state = past_key_values.conv_cache[self.layer_idx]

                # Clamp cache_position without reading its value
                # For numel() == 1, we know it's a single position, use the last slot
                cache_position_clamped = torch.clamp(cache_position, 0, self.L_cache - 1)

                conv_state = conv_state.roll(shifts=-1, dims=-1)
                # Use advanced indexing that doesn't require reading cache_position to CPU
                conv_state[:, :, cache_position_clamped] = Bx.to(
                    device=conv_state.device, dtype=conv_state.dtype
                )
                past_key_values.conv_cache[self.layer_idx].copy_(conv_state)

                conv_out = torch.sum(
                    conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1
                )
                if self.bias:
                    conv_out += self.conv.bias
                conv_out = conv_out.unsqueeze(-1)
            else:
                # Prefill phase or no cache
                if past_key_values is not None:
                    conv_state = torch.nn.functional.pad(
                        Bx, (self.L_cache - Bx.shape[-1], 0)
                    )
                    past_key_values.conv_cache[self.layer_idx].copy_(conv_state)
                conv_out = self.conv(Bx)[..., :seqlen]

            y = C * conv_out
            y = y.transpose(-1, -2).contiguous()
            y = self.out_proj(y)
            return y

        # Apply the patch
        Lfm2ShortConv.slow_forward = patched_slow_forward
        print("✅ Applied CUDA-graph compatibility patch to Lfm2ShortConv")

    def _initialize_cuda_graphs(self):
        """Initialize CUDA graphs for the decode phase."""
        print("🔧 Initializing CUDA graphs for decode phase...")

        # Apply monkey-patch to make LFM2 CUDA-graph compatible
        self._patch_lfm2_for_cuda_graphs()

        # Allocate static tensors for CUDA graph capture
        # Decode phase always has batch_size=1, seq_len=1
        self.static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.static_position_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.static_cache_position = torch.zeros((1,), dtype=torch.long, device=self.device)

        # Create a CUDA graph
        self.cuda_graph = torch.cuda.CUDAGraph()

        # Warmup: run the model once to initialize KV-cache structure
        # We need to run a full forward pass first to set up past_key_values
        print("   Warming up model for CUDA graph capture...")
        with torch.no_grad():
            # Create dummy input for warmup
            dummy_input = torch.tensor([[1]], dtype=torch.long, device=self.device)
            dummy_position = torch.tensor([[0]], dtype=torch.long, device=self.device)

            # First forward pass (prefill) to initialize KV cache
            outputs = self.model(
                input_ids=dummy_input,
                position_ids=dummy_position,
                use_cache=True,
                return_dict=True,
            )

            # Save the KV cache structure
            self.static_past_key_values = outputs.past_key_values

            # Second forward pass (decode) with KV cache - this is what we'll capture
            self.static_input_ids[0, 0] = 1
            self.static_position_ids[0, 0] = 1
            self.static_cache_position[0] = 1

            # Warmup for CUDA graph
            outputs = self.model(
                input_ids=self.static_input_ids,
                position_ids=self.static_position_ids,
                past_key_values=self.static_past_key_values,
                cache_position=self.static_cache_position,
                use_cache=True,
                return_dict=True,
            )

        # Capture the CUDA graph
        print("   Capturing CUDA graph...")
        with torch.cuda.graph(self.cuda_graph):
            self.static_outputs = self.model(
                input_ids=self.static_input_ids,
                position_ids=self.static_position_ids,
                past_key_values=self.static_past_key_values,
                cache_position=self.static_cache_position,
                use_cache=True,
                return_dict=True,
            )

        print("✅ CUDA graphs initialized successfully")

    def _generate_with_cuda_graphs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        eos_token_id: int,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Custom generation loop using CUDA graphs for the decode phase.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            eos_token_id: EOS token ID
            speaker_emb: Optional speaker embedding

        Returns:
            Generated token IDs [batch_size, seq_len + generated_len]
        """
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "CUDA graphs currently only support batch_size=1"

        # Prefill phase: process input sequence (variable length, cannot use CUDA graph)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                speaker_emb=speaker_emb,
                use_cache=True,
                return_dict=True,
            )

        # Get the KV cache from prefill
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Apply sampling to get first token
        next_token = self._sample_token(next_token_logits, temperature, top_p, repetition_penalty, input_ids)

        # Initialize generation
        generated_tokens = [next_token.item()]
        current_length = input_ids.shape[1]

        # Decode phase: generate tokens one by one using CUDA graph
        for step in range(max_new_tokens - 1):
            # Check for EOS
            if next_token.item() == eos_token_id:
                break

            # Update static tensors for CUDA graph replay
            self.static_input_ids[0, 0] = next_token
            self.static_position_ids[0, 0] = current_length + step
            self.static_cache_position[0] = current_length + step
            self.static_past_key_values = past_key_values

            # Replay CUDA graph
            self.cuda_graph.replay()

            # Get outputs from static tensors
            next_token_logits = self.static_outputs.logits[:, -1, :]
            past_key_values = self.static_outputs.past_key_values

            # Sample next token
            all_tokens = torch.cat([input_ids, torch.tensor([generated_tokens], device=self.device)], dim=1)
            next_token = self._sample_token(next_token_logits, temperature, top_p, repetition_penalty, all_tokens)
            generated_tokens.append(next_token.item())

        # Concatenate input and generated tokens
        generated_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=self.device)
        return torch.cat([input_ids, generated_tensor], dim=1)

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample next token from logits with temperature, top-p, and repetition penalty.

        Args:
            logits: Logits for next token [batch_size, vocab_size]
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            input_ids: Previous token IDs for repetition penalty

        Returns:
            Next token ID [1]
        """
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in input_ids[0].unique():
                logits[0, token_id] /= repetition_penalty

        # Apply temperature
        logits = logits / temperature

        # Apply top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(-1)

    def model_request(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          speaker_emb: Optional[torch.Tensor] = None,
                          token_callback: Optional[callable] = None) -> torch.Tensor:
        """
        Generate audio tokens from text tokens.

        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            speaker_emb: Optional speaker embedding [batch_size, speaker_emb_dim]
            token_callback: Optional callable(token_id: int) for streaming tokens
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Prepare kwargs for generation
        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': self.conf.max_new_tokens,
            'do_sample': True,
            'temperature': self.conf.temperature,
            'top_p': self.conf.top_p,
            'repetition_penalty': self.conf.repetition_penalty,
            'num_return_sequences': 1,
            'eos_token_id': self.player.end_of_speech,
        }

        # Add speaker_emb if provided
        if speaker_emb is not None:
            projection_dtype = self.model.model.speaker_emb_projection.weight.dtype
            speaker_emb = speaker_emb.to(self.device, dtype=projection_dtype)

        # Use inference engine if available
        if self.inference_engine is not None:
            with torch.no_grad():
                generated_ids = self.inference_engine.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=self.player.end_of_speech,
                    speaker_emb=speaker_emb,
                    token_callback=token_callback,
                )
        else:
            # Standard HuggingFace generation (no streaming callback support)
            if speaker_emb is not None:
                gen_kwargs['speaker_emb'] = speaker_emb

            with torch.no_grad():
                generated_ids = self.model.generate(**gen_kwargs)

        return generated_ids.to('cpu')

    def run_model(self, text: str, speaker_id: str = None, speaker_emb: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, str]:
        """
        Generate audio from text.

        Args:
            text: Input text to convert to speech
            speaker_id: Optional speaker ID (for multi-speaker models)
            speaker_emb: Optional speaker embedding tensor [1, speaker_emb_dim]
        """
        if (self.status == 'multispeaker') and (speaker_id is None):
            print('='*30)
            print('!!! YOU CAN CHOOSE A SPEAKER ID !!!')
            print(f'Speakers available:')
            print(print(*self.speaker_list, sep='\n'))
            print('='*30)
        elif (self.status == 'singlspeaker') and (speaker_id is not None):
            print('='*30)
            print('!!! This model does not support speaker selection !!!')
            print('='*30)

        input_ids, attention_mask = self.get_input_ids(text, speaker_id)

        t0 = time.perf_counter()
        model_output = self.model_request(input_ids, attention_mask, speaker_emb=speaker_emb)
        gen_time = time.perf_counter() - t0

        audio, _ = self.player.get_waveform(model_output)

        audio_duration = len(audio) / self.conf.sample_rate
        rtf = gen_time / audio_duration if audio_duration > 0 else float('inf')
        print(f"⏱  RTF: {rtf:.3f}x  (generated {audio_duration:.2f}s audio in {gen_time:.2f}s)")

        return audio, text
