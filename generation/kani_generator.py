"""Custom KaniTTS-based generation logic with async streaming via token callbacks"""

import asyncio
import time
import torch
import numpy as np
from typing import Optional

from kani_tts.core import TTSConfig, NemoAudioPlayer, KaniModel

from config import (
    MODEL_NAME, SAMPLE_RATE, TEMPERATURE, TOP_P, REPETITION_PENALTY, MAX_TOKENS,
    TEXT_VOCAB_SIZE, TOKENS_PER_FRAME, AUDIO_STEP,
    USE_LEARNABLE_ROPE, ALPHA_MIN, ALPHA_MAX, SPEAKER_EMB_DIM,
    ENABLE_SPEAKER_ADAPTERS, SPEAKER_ADAPTER_LAYERS, SPEAKER_ADAPTER_HIDDEN_DIM,
    USE_CUDA_GRAPHS, ATTN_IMPLEMENTATION, START_OF_SPEECH,
)


class KaniTTSGenerator:
    """Async wrapper around KaniTTS custom inference engine.

    Provides the same interface as VLLMTTSGenerator:
    - _generate_async(prompt_text, audio_writer, max_tokens)
    - generate_long_form_async(text, voice, player, ...)

    Uses asyncio.Lock + run_in_executor to bridge the synchronous
    KaniInferenceEngine to async FastAPI (batch_size=1 constraint).
    """

    def __init__(self):
        print(f"Loading KaniTTS custom engine: {MODEL_NAME}")

        config = TTSConfig(
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            text_vocab_size=TEXT_VOCAB_SIZE,
            tokens_per_frame=TOKENS_PER_FRAME,
            audio_step=AUDIO_STEP,
            use_learnable_rope=USE_LEARNABLE_ROPE,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            speaker_emb_dim=SPEAKER_EMB_DIM,
            enable_speaker_adapters=ENABLE_SPEAKER_ADAPTERS if ENABLE_SPEAKER_ADAPTERS else None,
            speaker_adapter_layers=SPEAKER_ADAPTER_LAYERS or None,
            speaker_adapter_hidden_dim=SPEAKER_ADAPTER_HIDDEN_DIM,
            use_cuda_graphs=USE_CUDA_GRAPHS,
            attn_implementation=ATTN_IMPLEMENTATION,
        )

        self.player = NemoAudioPlayer(config, text_tokenizer_name=MODEL_NAME)
        self.model = KaniModel(config, MODEL_NAME, self.player)
        self.config = config

        # Concurrency lock: batch_size=1 means only one generation at a time
        self._lock = asyncio.Lock()

        print("✅ KaniTTS custom engine ready")

    def prepare_input(self, prompt_text: str):
        """Build input_ids with special tokens (same format as KaniModel.get_input_ids).

        Returns list of token IDs for token counting.
        """
        input_ids, _ = self.model.get_input_ids(prompt_text)
        return input_ids[0].tolist()

    async def _generate_async(self, prompt_text: str, audio_writer, max_tokens: int = MAX_TOKENS,
                               speaker_emb: Optional[torch.Tensor] = None,
                               temperature: Optional[float] = None,
                               top_p: Optional[float] = None,
                               repetition_penalty: Optional[float] = None):
        """Async generation with token-level streaming to audio_writer.

        Args:
            prompt_text: Text to synthesize
            audio_writer: StreamingAudioWriter instance to receive tokens
            max_tokens: Maximum tokens to generate
            speaker_emb: Optional speaker embedding tensor

        Returns:
            Dictionary with generation metrics
        """
        async with self._lock:
            loop = asyncio.get_event_loop()

            point_1 = time.time()

            # Token tracking
            all_token_ids = []
            audio_token_count = 0
            inside_speech = False

            def token_callback(token_id: int):
                """Called synchronously from the inference engine for each generated token."""
                nonlocal audio_token_count, inside_speech
                all_token_ids.append(token_id)
                audio_writer.add_token(token_id)

                if token_id == self.player.start_of_speech:
                    inside_speech = True
                elif token_id == self.player.end_of_speech:
                    inside_speech = False
                elif inside_speech:
                    audio_token_count += 1

            def run_generation():
                """Synchronous generation in executor thread."""
                input_ids, attention_mask = self.model.get_input_ids(prompt_text)

                # The custom engine includes START_OF_SPEECH in prefill input,
                # but StreamingAudioWriter expects to see it as a generated token.
                # Manually send it before generation starts.
                token_callback(START_OF_SPEECH)

                self.model.model_request(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    speaker_emb=speaker_emb,
                    token_callback=token_callback,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

            await loop.run_in_executor(None, run_generation)

            point_2 = time.time()
            generation_time = point_2 - point_1

            # Calculate metrics
            FRAMES_PER_SECOND = 12.5
            num_frames = audio_token_count // TOKENS_PER_FRAME
            audio_duration = num_frames / FRAMES_PER_SECOND
            rtf = generation_time / audio_duration if audio_duration > 0 else 0

            input_ids_list = self.prepare_input(prompt_text)
            prompt_tokens = len(input_ids_list)
            generated_tokens = len(all_token_ids)

            print(f"\n[KaniTTS] Generation complete. Prompt tokens: {prompt_tokens}, Generated tokens: {generated_tokens}")
            print(f"          Audio tokens: {audio_token_count}, Frames: {num_frames}, Audio duration: {audio_duration:.2f}s")
            print(f"          Generation time: {generation_time:.2f}s, RTF: {rtf:.3f}")

            return {
                'all_token_ids': all_token_ids,
                'generation_time': generation_time,
                'audio_duration': audio_duration,
                'rtf': rtf,
                'point_1': point_1,
                'point_2': point_2,
            }

    async def generate_long_form_async(self, text, player, max_chunk_duration=12.0,
                                       silence_duration=0.2, max_tokens=MAX_TOKENS,
                                       speaker_emb=None, temperature: Optional[float] = None,
                                       top_p: Optional[float] = None,
                                       repetition_penalty: Optional[float] = None,
                                       ref_text: Optional[str] = None):
        """Generate speech for long text by splitting into chunks."""
        from generation.chunking import split_into_sentences, estimate_duration
        from audio.streaming import StreamingAudioWriter
        from config import CHUNK_SIZE, LOOKBACK_FRAMES

        estimated_duration = estimate_duration(text)
        print(f"\n[Long-form] Estimated duration: {estimated_duration:.1f}s for text length: {len(text)} chars")

        chunks = split_into_sentences(text, max_duration_seconds=max_chunk_duration)
        print(f"[Long-form] Split into {len(chunks)} chunks")

        audio_segments = []
        chunks_info = []
        total_generation_time = 0

        for i, chunk in enumerate(chunks):
            print(f"\n[Long-form] Generating chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
            prompt_text = chunk if not ref_text or not ref_text.strip() else (
                f"Reference text: {ref_text.strip()}\nTarget text: {chunk}"
            )

            audio_writer = StreamingAudioWriter(
                player,
                output_file=None,
                chunk_size=CHUNK_SIZE,
                lookback_frames=LOOKBACK_FRAMES,
            )
            audio_writer.start()

            result = await self._generate_async(
                prompt_text,
                audio_writer,
                max_tokens=max_tokens,
                speaker_emb=speaker_emb,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            audio = audio_writer.finalize()

            if audio is not None and len(audio) > 0:
                audio_segments.append(audio)
                chunks_info.append({
                    'chunk_index': i,
                    'text': chunk,
                    'duration': result['audio_duration'],
                    'generation_time': result['generation_time'],
                    'rtf': result['rtf'],
                })
                total_generation_time += result['generation_time']
            else:
                print(f"[Long-form] Warning: No audio generated for chunk {i+1}")

        if len(audio_segments) == 0:
            raise ValueError("No audio was generated")

        if len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            final_audio = self._concatenate_with_silence(audio_segments, silence_duration)

        total_duration = len(final_audio) / SAMPLE_RATE

        print(f"\n[Long-form] Complete!")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Total generation time: {total_generation_time:.2f}s")
        print(f"  Overall RTF: {total_generation_time / total_duration:.3f}")

        return {
            'audio': final_audio,
            'chunks_info': chunks_info,
            'total_duration': total_duration,
            'total_generation_time': total_generation_time,
            'num_chunks': len(chunks),
        }

    def _concatenate_with_silence(self, audio_segments, silence_duration=0.2):
        """Concatenate audio segments with short silence between them."""
        if len(audio_segments) == 1:
            return audio_segments[0]

        silence_samples = int(silence_duration * SAMPLE_RATE)
        silence = np.zeros(silence_samples, dtype=audio_segments[0].dtype)

        result = audio_segments[0]
        for next_segment in audio_segments[1:]:
            result = np.concatenate([result, silence, next_segment])
        return result
