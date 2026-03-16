"""Configuration and constants for Kani TTS"""

import os

# Tokenizer configuration
TOKENIZER_LENGTH = 64400

# Special tokens
START_OF_TEXT = 1
END_OF_TEXT = 2
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7
AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10

# Audio configuration
CODEBOOK_SIZE = 4032
SAMPLE_RATE = 22050

# Streaming configuration
CHUNK_SIZE = 25  # Number of new frames to output per iteration
LOOKBACK_FRAMES = 15  # Number of frames to include from previous context

# Generation configuration
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2
REPETITION_CONTEXT_SIZE = 20
MAX_TOKENS = 3000

# Long-form generation configuration
LONG_FORM_THRESHOLD_SECONDS = 60.0  # Auto-enable chunking for texts estimated >15s
LONG_FORM_CHUNK_DURATION = 30.0     # Target duration per chunk (stay within 5-15s training distribution)
LONG_FORM_SILENCE_DURATION = 0.1    # Silence between chunks in seconds

# BemaTTS configuration
TEXT_VOCAB_SIZE = 64400
TOKENS_PER_FRAME = 4
AUDIO_STEP = 1.0
USE_LEARNABLE_ROPE = True
ALPHA_MIN = 0.1
ALPHA_MAX = 2.0
SPEAKER_EMB_DIM = 128
USE_CUDA_GRAPHS = True
ATTN_IMPLEMENTATION = "sdpa"

# Model paths
# Override MODEL_NAME via env var MODEL_NAME if supplied
MODEL_NAME = os.getenv("MODEL_NAME", "shiprocket-ai/kani-tts-2-hindi-95000")
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
