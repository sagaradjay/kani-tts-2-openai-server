"""FastAPI server for Kani TTS with streaming support"""

import io
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
import torch
from scipy.io.wavfile import write as wav_write
import base64
import json

from audio import StreamingAudioWriter
from generation.kani_generator import KaniTTSGenerator
from pathlib import Path
from config import CHUNK_SIZE, LOOKBACK_FRAMES, TEMPERATURE, TOP_P, REPETITION_PENALTY, MAX_TOKENS, LONG_FORM_THRESHOLD_SECONDS, LONG_FORM_SILENCE_DURATION, LONG_FORM_CHUNK_DURATION
from speaker_embedder import SpeakerEmbedder

SPEAKERS_DIR = Path(__file__).parent / "speakers"
VOICES_DIR = Path(__file__).parent / "voices"
VOICE_REF_TEXTS_DIR = Path(__file__).parent / "voice_ref_texts"

# AUTH_TOKEN: when set (non-empty), API requires Authorization: Bearer <token>
# Pass at runtime via -e AUTH_TOKEN=... ; if not set, defaults to "" (no auth)
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()


app = FastAPI(title="Kani TTS API", version="1.0.0")

# Add CORS middleware to allow client.html to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require Authorization: Bearer <AUTH_TOKEN> when AUTH_TOKEN is set."""
    if AUTH_TOKEN:
        # Skip auth for health and root
        if request.url.path in ("/", "/health"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:].strip() != AUTH_TOKEN:
            return Response(
                content='{"detail":"Missing or invalid Authorization"}',
                status_code=401,
                media_type="application/json",
            )
    return await call_next(request)

# Global instances (initialized on startup)
generator = None
player = None
speaker_embedder = None
speaker_embeddings: dict[str, torch.Tensor] = {}


def build_prompt_text(target_text: str, ref_text: Optional[str] = None) -> str:
    """Serialize prompt text in the same format used by cloning-stage training."""
    target_text = target_text.strip()
    if ref_text and ref_text.strip():
        return f"Reference text: {ref_text.strip()}\nTarget text: {target_text}"
    return target_text


@lru_cache(maxsize=1)
def load_shared_voice_ref_texts() -> dict[str, str]:
    """Load voice -> ref_text mappings from voices/ref_text.txt."""
    shared_ref_path = VOICES_DIR / "ref_text.txt"
    mappings: dict[str, str] = {}

    if not shared_ref_path.exists():
        return mappings

    for raw_line in shared_ref_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, text = line.partition(",")
        if not sep:
            continue
        voice_name = name.strip()
        ref_text = text.strip()
        if voice_name and ref_text:
            mappings[voice_name] = ref_text

    return mappings


def load_voice_ref_text(voice_name: str) -> Optional[str]:
    """Load reference text from voices/ref_text.txt or legacy voice_ref_texts/<voice>.txt."""
    shared_mappings = load_shared_voice_ref_texts()
    if voice_name in shared_mappings:
        return shared_mappings[voice_name]

    # Legacy fallback
    ref_text_path = VOICE_REF_TEXTS_DIR / f"{voice_name}.txt"
    if not ref_text_path.exists():
        return None

    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    return ref_text or None


def compute_speaker_similarity(reference_emb: Optional[torch.Tensor], generated_audio: np.ndarray) -> Optional[float]:
    """Compute cosine similarity between the reference speaker embedding and generated audio embedding."""
    if speaker_embedder is None or reference_emb is None or generated_audio is None or generated_audio.size == 0:
        return None

    try:
        generated_emb = speaker_embedder.embed_audio(generated_audio, sample_rate=22050)
        ref_vec = reference_emb.detach().float().cpu().view(1, -1)
        gen_vec = generated_emb.detach().float().cpu().view(1, -1)
        return torch.nn.functional.cosine_similarity(ref_vec, gen_vec, dim=-1).item()
    except Exception as e:
        print(f"[Server] Failed to compute speaker similarity: {e}")
        return None


def log_speaker_similarity(requested_voice: str, similarity: Optional[float]):
    """Log speaker similarity if available."""
    if similarity is None:
        return
    voice_label = requested_voice or "random"
    print(f"[Server] Speaker similarity for '{voice_label}': {similarity:.4f}")


def build_voice_embeddings():
    """
    On startup, scan the voices directory for audio files and create
    missing speaker embeddings in the speakers directory.

    This only creates .pt files that don't already exist, so any
    manually curated embeddings are left untouched.
    """
    if not VOICES_DIR.exists():
        return

    SPEAKERS_DIR.mkdir(exist_ok=True, parents=True)
    VOICE_REF_TEXTS_DIR.mkdir(exist_ok=True, parents=True)

    # Instantiate embedder once and reuse for all voices
    embedder = SpeakerEmbedder()

    created = []

    for audio_file in sorted(VOICES_DIR.glob("*.wav")):
        name = audio_file.stem  # e.g. "male_1"
        out_path = SPEAKERS_DIR / f"{name}.pt"

        # Skip if an embedding already exists
        if out_path.exists():
            continue

        try:
            print(f"🎙  Computing speaker embedding for '{name}' from {audio_file} ...")
            emb = embedder.embed_audio_file(str(audio_file))  # [1, 128]
            emb_vec = emb[0].cpu()  # [128]
            torch.save(emb_vec, out_path)
            created.append(name)
        except Exception as e:
            print(f"⚠️  Failed to build embedding for {audio_file}: {e}")

    if created:
        print(f"✅ Created embeddings for voices: {created}")
    else:
        print("ℹ️  No new voice embeddings created (all up to date).")


def load_speaker_embeddings():
    """Load all speaker embedding .pt files from the speakers directory."""
    for pt_file in sorted(SPEAKERS_DIR.glob("*.pt")):
        name = pt_file.stem  # e.g. "speaker_1"
        emb = torch.load(pt_file, weights_only=True).unsqueeze(0)  # [128] -> [1, 128]
        speaker_embeddings[name] = emb
    print(f"Loaded {len(speaker_embeddings)} speaker embeddings: {list(speaker_embeddings.keys())}")


class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    top_p: Optional[float] = TOP_P
    chunk_size: Optional[int] = CHUNK_SIZE
    lookback_frames: Optional[int] = LOOKBACK_FRAMES


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech request model"""
    input: str = Field(..., description="Text to convert to speech")
    ref_text: Optional[str] = Field(default=None, description="Optional reference transcript paired with the reference voice audio/embedding")
    model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] = Field(default="tts-1", description="TTS model to use")
    voice: str = Field(default="random", description="Speaker name matching a .pt file in /speakers (e.g. 'speaker_1'). Use 'random' or omit the field to skip speaker embedding.")
    response_format: Literal["wav", "pcm"] = Field(default="wav", description="Audio format: wav or pcm")
    stream_format: Optional[Literal["sse", "audio"]] = Field(default=None, description="Use 'sse' for Server-Sent Events streaming")
    temperature: Optional[float] = Field(default=TEMPERATURE, description="Sampling temperature")
    top_p: Optional[float] = Field(default=TOP_P, description="Top-p nucleus sampling")
    repetition_penalty: Optional[float] = Field(default=REPETITION_PENALTY, description="Penalty for recently used tokens")
    max_tokens: Optional[int] = Field(default=MAX_TOKENS, description="Maximum audio tokens to generate")
    # Long-form generation parameters
    enable_long_form: Optional[bool] = Field(default=True, description="Auto-detect and use long-form generation for texts >15s")
    max_chunk_duration: Optional[float] = Field(default=12.0, description="Max duration per chunk in long-form mode (seconds)")
    silence_duration: Optional[float] = Field(default=0.2, description="Silence between chunks in long-form mode (seconds)")


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global generator, player, speaker_embedder
    print("🚀 Initializing KaniTTS custom engine...")

    # First, build any missing embeddings from raw voice files
    build_voice_embeddings()
    VOICE_REF_TEXTS_DIR.mkdir(exist_ok=True, parents=True)
    load_shared_voice_ref_texts.cache_clear()

    load_speaker_embeddings()
    speaker_embedder = SpeakerEmbedder()

    generator = KaniTTSGenerator()

    # Reuse the NemoAudioPlayer from the custom engine (avoids duplicate codec on GPU)
    player = generator.player
    print("✅ KaniTTS models initialized successfully!")


@app.get("/health")
async def health_check():
    """Check if server is ready"""
    return {
        "status": "healthy",
        "tts_initialized": generator is not None and player is not None
    }


@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAISpeechRequest):
    """OpenAI-compatible speech generation endpoint

    Supports both streaming (SSE) and non-streaming modes:
    - Without stream_format: Returns complete audio file (WAV or PCM)
    - With stream_format="sse": Returns Server-Sent Events with audio chunks
    """
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    # Look up speaker embedding only when the caller explicitly selected a voice.
    speaker_emb = None
    requested_voice = request.voice.strip()
    if requested_voice and requested_voice.lower() != "random":
        if requested_voice not in speaker_embeddings:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown voice '{requested_voice}'. Available: {list(speaker_embeddings.keys())}"
            )
        speaker_emb = speaker_embeddings[requested_voice]

    resolved_ref_text = request.ref_text
    if (not resolved_ref_text or not resolved_ref_text.strip()) and requested_voice:
        resolved_ref_text = load_voice_ref_text(requested_voice)

    # Prepare prompt text (no voice prefix — speaker embedding handles identity)
    prompt_text = build_prompt_text(request.input, resolved_ref_text)

    # Streaming mode (SSE)
    if request.stream_format == "sse":
        async def sse_generator():
            """Generate Server-Sent Events with audio chunks"""
            import asyncio
            import queue as thread_queue
            from generation.chunking import estimate_duration, split_into_sentences

            chunk_queue = thread_queue.Queue()

            # Estimate duration to determine if we need long-form generation
            estimated_duration = estimate_duration(request.input)
            use_long_form = estimated_duration > LONG_FORM_THRESHOLD_SECONDS

            # Track token counts for usage reporting
            input_token_count = 0
            output_token_count = 0

            if use_long_form:
                # Long-form streaming: stream each sentence chunk as it's generated
                print(f"[Server] Using long-form SSE streaming (estimated {estimated_duration:.1f}s)")

                async def generate_async_long_form():
                    nonlocal input_token_count, output_token_count
                    try:
                        # Split into chunks
                        chunks = split_into_sentences(request.input, max_duration_seconds=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION)
                        total_chunks = len(chunks)

                        for i, text_chunk in enumerate(chunks):
                            # Custom list wrapper that pushes chunks to queue
                            class ChunkList(list):
                                def append(self, chunk):
                                    super().append(chunk)
                                    chunk_queue.put(("chunk", chunk))

                            audio_writer = StreamingAudioWriter(
                                player,
                                output_file=None,
                                chunk_size=CHUNK_SIZE,
                                lookback_frames=LOOKBACK_FRAMES
                            )
                            audio_writer.audio_chunks = ChunkList()
                            audio_writer.start()

                            chunk_prompt_text = build_prompt_text(text_chunk, resolved_ref_text)
                            result = await generator._generate_async(
                                chunk_prompt_text,
                                audio_writer,
                                max_tokens=request.max_tokens or MAX_TOKENS,
                                speaker_emb=speaker_emb,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                repetition_penalty=request.repetition_penalty,
                            )
                            audio_writer.finalize()

                            # Track tokens
                            input_token_count += len(generator.prepare_input(chunk_prompt_text))
                            output_token_count += len(result.get('all_token_ids', []))

                            # Add silence between chunks (except after last chunk)
                            if i < total_chunks - 1:
                                silence_samples = int((request.silence_duration or LONG_FORM_SILENCE_DURATION) * 22050)
                                silence = np.zeros(silence_samples, dtype=np.float32)
                                chunk_queue.put(("chunk", silence))

                        chunk_queue.put(("done", {"input": input_token_count, "output": output_token_count}))
                    except Exception as e:
                        print(f"Generation error: {e}")
                        import traceback
                        traceback.print_exc()
                        chunk_queue.put(("error", str(e)))

                gen_task = asyncio.create_task(generate_async_long_form())
            else:
                # Standard streaming for short texts
                print(f"[Server] Using standard SSE streaming (estimated {estimated_duration:.1f}s)")

                # Custom list wrapper that pushes chunks to queue
                class ChunkList(list):
                    def append(self, chunk):
                        super().append(chunk)
                        chunk_queue.put(("chunk", chunk))

                audio_writer = StreamingAudioWriter(
                    player,
                    output_file=None,
                    chunk_size=CHUNK_SIZE,
                    lookback_frames=LOOKBACK_FRAMES
                )
                audio_writer.audio_chunks = ChunkList()

                # Start generation in background task
                async def generate_async():
                    nonlocal input_token_count, output_token_count
                    try:
                        audio_writer.start()
                        result = await generator._generate_async(
                            prompt_text,
                            audio_writer,
                            max_tokens=request.max_tokens or MAX_TOKENS,
                            speaker_emb=speaker_emb,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            repetition_penalty=request.repetition_penalty,
                        )
                        audio_writer.finalize()

                        # Extract token counts from result
                        input_token_count = len(generator.prepare_input(prompt_text))
                        output_token_count = len(result.get('all_token_ids', []))

                        chunk_queue.put(("done", {"input": input_token_count, "output": output_token_count}))
                    except Exception as e:
                        print(f"Generation error: {e}")
                        import traceback
                        traceback.print_exc()
                        chunk_queue.put(("error", str(e)))

                # Start generation as async task
                gen_task = asyncio.create_task(generate_async())

            # Stream chunks as they arrive
            try:
                while True:
                    msg_type, data = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: chunk_queue.get(timeout=30)
                    )

                    if msg_type == "chunk":
                        # Convert numpy array to int16 PCM
                        pcm_data = (data * 32767).astype(np.int16)

                        # Encode as base64
                        audio_base64 = base64.b64encode(pcm_data.tobytes()).decode('utf-8')

                        # Send SSE event: speech.audio.delta
                        event_data = {
                            "type": "speech.audio.delta",
                            "audio": audio_base64
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"

                    elif msg_type == "done":
                        # Send SSE event: speech.audio.done with usage stats
                        token_counts = data
                        similarity = None
                        if requested_voice and requested_voice.lower() != "random" and audio_writer.audio_chunks:
                            try:
                                full_audio = np.concatenate(audio_writer.audio_chunks)
                            except Exception:
                                full_audio = None
                            if full_audio is not None:
                                similarity = compute_speaker_similarity(speaker_emb, full_audio)
                                log_speaker_similarity(requested_voice, similarity)
                        event_data = {
                            "type": "speech.audio.done",
                            "usage": {
                                "input_tokens": token_counts["input"],
                                "output_tokens": token_counts["output"],
                                "total_tokens": token_counts["input"] + token_counts["output"]
                            },
                            "speaker_similarity": similarity,
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                        break

                    elif msg_type == "error":
                        # Send error event
                        error_data = {
                            "type": "error",
                            "error": data
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                        break

            finally:
                await gen_task

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming mode (complete audio file)
    else:
        try:
            # Estimate duration to determine if we need long-form generation
            from generation.chunking import estimate_duration
            estimated_duration = estimate_duration(request.input)

            # Use long-form generation for longer texts
            use_long_form = estimated_duration > LONG_FORM_THRESHOLD_SECONDS

            if use_long_form:
                print(f"[Server] Using long-form generation (estimated {estimated_duration:.1f}s)")
                result = await generator.generate_long_form_async(
                    text=request.input,
                    player=player,
                    max_chunk_duration=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION,
                    silence_duration=request.silence_duration or LONG_FORM_SILENCE_DURATION,
                    max_tokens=request.max_tokens or MAX_TOKENS,
                    speaker_emb=speaker_emb,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    ref_text=resolved_ref_text,
                )
                full_audio = result['audio']
            else:
                # Standard generation for short texts
                print(f"[Server] Using standard generation (estimated {estimated_duration:.1f}s)")
                audio_writer = StreamingAudioWriter(
                    player,
                    output_file=None,
                    chunk_size=CHUNK_SIZE,
                    lookback_frames=LOOKBACK_FRAMES
                )
                audio_writer.start()

                # Generate speech
                result = await generator._generate_async(
                    prompt_text,
                    audio_writer,
                    max_tokens=request.max_tokens or MAX_TOKENS,
                    speaker_emb=speaker_emb,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                )

                # Finalize and get audio
                audio_writer.finalize()

                if not audio_writer.audio_chunks:
                    raise HTTPException(status_code=500, detail="No audio generated")

                # Concatenate all chunks
                full_audio = np.concatenate(audio_writer.audio_chunks)

            similarity = None
            if requested_voice and requested_voice.lower() != "random":
                similarity = compute_speaker_similarity(speaker_emb, full_audio)
                log_speaker_similarity(requested_voice, similarity)

            # Return based on response_format
            if request.response_format == "pcm":
                # Return raw PCM (int16)
                pcm_data = (full_audio * 32767).astype(np.int16)
                return Response(
                    content=pcm_data.tobytes(),
                    media_type="application/octet-stream",
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Sample-Rate": "22050",
                        "X-Channels": "1",
                        "X-Bit-Depth": "16",
                        "X-Speaker-Similarity": f"{similarity:.4f}" if similarity is not None else "",
                    }
                )
            else:  # wav
                # Convert to WAV bytes
                wav_buffer = io.BytesIO()
                wav_write(wav_buffer, 22050, full_audio)
                wav_buffer.seek(0)

                return Response(
                    content=wav_buffer.read(),
                    media_type="audio/wav",
                    headers={
                        "X-Speaker-Similarity": f"{similarity:.4f}" if similarity is not None else "",
                    }
                )

        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Kani TTS API",
        "version": "1.0.0",
        "endpoints": {
            "/v1/audio/speech": "POST - OpenAI-compatible speech generation",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("🎤 Starting Kani TTS Server (custom engine)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
