"""Text-to-speech generation modules"""

from .kani_generator import KaniTTSGenerator
from .chunking import split_into_sentences, estimate_duration

__all__ = ['KaniTTSGenerator', 'split_into_sentences', 'estimate_duration']
