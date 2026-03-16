"""
Kani-TTS: Text-to-Speech using Neural Audio Codec

A simple interface for generating speech from text using a causal language model
and NVIDIA NeMo audio codec.
"""

from .api import KaniTTS, suppress_all_logs
from .core import TTSConfig
from .speaker_embedder import SpeakerEmbedder, compute_speaker_embedding

__version__ = "0.1.0"
__all__ = ["KaniTTS", "TTSConfig", "suppress_all_logs", "SpeakerEmbedder", "compute_speaker_embedding"]
