# KaniTTS

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance Text-to-Speech (TTS) system with a custom inference engine, providing an OpenAI-compatible API for fast, streaming speech generation with speaker embedding support.

## Features

- **Custom Inference Engine**: Optimized decode loop with Triton kernels, CUDA graphs, and static KV cache
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- **Real-Time Streaming**: Server-Sent Events (SSE) with token-level audio streaming
- **Long-Form Generation**: Automatic text chunking for generating speech from lengthy inputs
- **Speaker Embeddings**: 128-dimensional speaker embeddings for voice identity control
- **BemaTTS**: Frame-level position encoding with learnable RoPE
- **Flexible Output Formats**: WAV, PCM, or streaming SSE

## Architecture

```
FastAPI Server (OpenAI-compatible endpoint)
            |
KaniTTS Custom Inference Engine (CUDA graphs + Triton kernels)
            |
Token-level Streaming + NeMo NanoCodec Decoder
            |
Output: WAV / PCM / Server-Sent Events
```

The system uses:
- **TTS Model**: `nineninesix/kani-tts-2-pt` | `nineninesix/kani-tts-2-en`
- **Audio Codec**: `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps` (12.5fps, 4 codebooks)
- **Inference Engine**: Custom engine with CUDA graphs, Triton fused kernels, and static cache
- **Sample Rate**: 22050 Hz, 16-bit, mono

## Installation

### Prerequisites
- Linux
- Python 3.10 -- 3.12
- NVIDIA GPU with CUDA 12.8+ and bf16 support (Ampere or newer, e.g. A10, RTX 3090+)

### Install Dependencies

1. Create and activate virtual environment:
```bash
cd <your_project_dir>
python -m venv venv
source venv/bin/activate
```

2. Install FastAPI and server dependencies:
```bash
pip install fastapi uvicorn scipy
```

3. Install nemo-toolkit:
```bash
pip install "nemo-toolkit[tts]==2.4.0"
```

4. Upgrade transformers (required for model compatibility):
```bash
pip install "transformers==4.57.1"
```

5. Install Triton (required for fused kernels):
```bash
pip install triton
```

**Known issues**

- `nemo-toolkit[tts]` requires `transformers==4.53`, but this project requires `transformers==4.57.1` for model compatibility. Install nemo-toolkit first, then upgrade transformers.

- `nemo-toolkit[tts]` requires `ffmpeg`. Install it with `apt install ffmpeg` if not already present.

- For Blackwell GPUs `nemo-toolkit[tts]==2.5.1` works too.

## Quick Start

### Start the Server

```bash
python server.py
```

The server will start on `http://localhost:8000` and automatically download the required models on first run.

### Check Server Health

```bash
curl http://localhost:8000/health
```

### Generate Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of the text to speech system.",
    "voice": "speaker_1",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Generate Speech (Streaming)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This will be streamed in real-time as audio chunks.",
    "voice": "speaker_3",
    "stream_format": "sse"
  }'
```

### Frontend

You can test the API with [open-audio](https://github.com/nineninesix-ai/open-audio), a Next.js frontend that connects to this server out of the box.

## API Reference

### POST `/v1/audio/speech`

OpenAI-compatible endpoint for text-to-speech generation.

#### Request Body

```json
{
  "input": "Text to convert to speech",
  "model": "tts-1",
  "voice": "speaker_1",
  "response_format": "wav",
  "stream_format": null,
  "max_chunk_duration": 30.0,
  "silence_duration": 0.2
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | *required* | Text to convert to speech |
| `model` | string | `"tts-1"` | OpenAI compatibility field (no effect on model selection) |
| `voice` | string | `"speaker_1"` | Speaker name matching a `.pt` file in `speakers/`. Use `"random"` to skip speaker embedding. |
| `response_format` | string | `"wav"` | `"wav"` or `"pcm"` |
| `stream_format` | string | `null` | `null` for complete file, `"sse"` for streaming |
| `max_chunk_duration` | float | `30.0` | Max seconds per chunk in long-form mode |
| `silence_duration` | float | `0.2` | Silence between chunks in long-form mode |

#### Available Voices

Speaker embeddings are stored as `.pt` files in the `speakers/` directory:

- `speaker_1` through `speaker_10`

To add a new voice, place a 128-dimensional speaker embedding tensor as a `.pt` file in `speakers/`. All embeddings are loaded at server startup.

#### Response Formats

**Non-Streaming** (`stream_format` is null):
- `wav` - Complete WAV file (default)
- `pcm` - Raw PCM audio with metadata headers (`X-Sample-Rate`, `X-Channels`, `X-Bit-Depth`)

**Streaming** (`stream_format: "sse"`):
- Server-Sent Events with base64-encoded PCM audio chunks

#### Streaming Event Format (SSE)

```
data: {"type": "speech.audio.delta", "audio": "<base64_pcm_chunk>"}
data: {"type": "speech.audio.delta", "audio": "<base64_pcm_chunk>"}
data: {"type": "speech.audio.done", "usage": {"input_tokens": 25, "output_tokens": 487, "total_tokens": 512}}
```

### GET `/health`

Returns server and model status.

```json
{
  "status": "healthy",
  "tts_initialized": true
}
```

## Long-Form Generation

For texts estimated to take more than 40 seconds to speak, the system automatically:

1. Splits text into sentence-based chunks (~30 seconds each)
2. Generates each chunk independently with the same speaker embedding
3. Concatenates audio segments with configurable silence
4. Returns seamless combined audio

Control long-form behavior:
```json
{
  "input": "Very long text...",
  "voice": "speaker_1",
  "max_chunk_duration": 30.0,
  "silence_duration": 0.2
}
```

## Configuration

Key configuration parameters in [config.py](config.py):

```python
# Audio Settings
SAMPLE_RATE = 22050
CHUNK_SIZE = 25                        # Frames per streaming chunk
LOOKBACK_FRAMES = 15                   # Context frames for decoding

# Generation Parameters
TEMPERATURE = 1.0
TOP_P = 0.95
REPETITION_PENALTY = 1.1
MAX_TOKENS = 3000

# Long-Form Settings
LONG_FORM_THRESHOLD_SECONDS = 40.0
LONG_FORM_CHUNK_DURATION = 30.0
LONG_FORM_SILENCE_DURATION = 0.2

# BemaTTS
TOKENS_PER_FRAME = 4
AUDIO_STEP = 1.0
USE_LEARNABLE_ROPE = True
SPEAKER_EMB_DIM = 128
USE_CUDA_GRAPHS = True

# Models
MODEL_NAME = "nineninesix/kani-tts-2-pt"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
```

## Speaker Embeddings

The system uses 128-dimensional speaker embeddings for voice identity. Embeddings are pre-saved as PyTorch tensors in `speakers/`.

### Adding a New Speaker

1. Generate a 128-dim embedding from a reference audio using a speaker encoder (e.g. `nineninesix/speaker-emb-tbr`)
2. Save it as a `.pt` file:
```python
import torch
embedding = torch.tensor([...])  # 128-dim vector
torch.save(embedding, './speakers/my_speaker.pt')
```
3. Restart the server. The new voice is available as `"voice": "my_speaker"`

## Project Structure
```
├── server.py                  # FastAPI application and main entry point
├── config.py                  # Configuration and constants
├── speakers/                  # Pre-saved speaker embedding .pt files
│   ├── speaker_1.pt
│   ├── speaker_2.pt
│   └── ...
├── audio/
│   ├── __init__.py
│   └── streaming.py           # Streaming audio writer with sliding window decoder
├── generation/
│   ├── __init__.py
│   ├── kani_generator.py      # Async wrapper around custom inference engine
│   └── chunking.py            # Text splitting for long-form generation
└── kani_tts/                  # Custom inference engine
    ├── __init__.py
    ├── api.py                 # Simple KaniTTS API (for standalone use)
    ├── core.py                # TTSConfig, NemoAudioPlayer, KaniModel
    ├── model.py               # BemaTTS model with frame-level position encoding
    ├── inference_engine.py    # Optimized decode loop with CUDA graphs
    ├── optimized_decode.py    # Fused decoder operations
    ├── triton_kernels.py      # Fused RMSNorm, SiLU-mul, RoPE kernels
    ├── static_cache.py        # Static KV cache for CUDA graph compatibility
    ├── context.py             # Thread-local context for CUDA graph capture
    └── speaker_embedder.py    # WavLM-based speaker embedding extraction
```

## Troubleshooting

### Audio Quality Issues

1. Ensure sample rate matches (22050 Hz)
2. For long-form, adjust chunk duration:
   ```json
   {"max_chunk_duration": 20.0}
   ```
3. Increase lookback frames for smoother transitions in [config.py](config.py):
   ```python
   LOOKBACK_FRAMES = 20
   ```

### Model Download Issues

Models are automatically downloaded from HuggingFace on first run. If downloads fail:
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('nineninesix/kani-tts-2-pt')
AutoModelForCausalLM.from_pretrained('nineninesix/kani-tts-2-pt')
"
```

## License

The code in this repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Model weights and codecs are subject to their own licenses:

- **nineninesix/kani-tts-2-pt** weights: [Liquid AI LFM License](https://www.liquid.ai/lfm-license)
- **NeMo Nano Codec**: [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)

## Support

For issues, questions, or feature requests, please open an issue on GitHub or [Discord](https://discord.gg/NzP3rjB4SB)
