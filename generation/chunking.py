"""Text chunking utilities for long-form speech generation"""

import re
from typing import List


def split_into_sentences(text: str, max_duration_seconds: float = 12.0) -> List[str]:
    """Split text into sentences suitable for TTS generation

    The chunking strategy ensures each chunk is within the model's training
    distribution (5-15 seconds of speech) for optimal quality.

    Args:
        text: Input text to split
        max_duration_seconds: Maximum target duration per chunk (default 12s)

    Returns:
        List of text chunks, each representing ~max_duration_seconds of speech

    Notes:
        - Uses heuristic of ~15 characters per second of speech
        - Splits on sentence boundaries (., !, ?)
        - Keeps sentences together when possible
        - Fallback to word-level splitting for very long sentences
    """
    # Heuristic: ~15 characters per second of speech (adjustable based on your model)
    max_chars = int(max_duration_seconds * 15)

    # Split into sentences using common punctuation
    # This regex keeps the punctuation with the sentence
    sentence_pattern = r'([.!?]+[\s\n]+|[.!?]+$)'
    parts = re.split(sentence_pattern, text)

    # Reconstruct sentences (combine text + punctuation)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]
        sentences.append(sentence.strip())

    # Handle last part if no punctuation at end
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    # Filter empty sentences
    sentences = [s for s in sentences if s]

    # Group sentences into chunks
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If single sentence exceeds max, split it by words
        if len(sentence) > max_chars:
            # Save current chunk if any
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Split long sentence into word-based chunks
            words = sentence.split()
            word_chunk = ""
            for word in words:
                if len(word_chunk) + len(word) + 1 <= max_chars:
                    word_chunk += word + " "
                else:
                    chunks.append(word_chunk.strip())
                    word_chunk = word + " "

            if word_chunk.strip():
                current_chunk = word_chunk.strip()

        # Check if adding this sentence would exceed max
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def estimate_duration(text: str, chars_per_second: float = 15.0) -> float:
    """Estimate speech duration for given text

    Args:
        text: Input text
        chars_per_second: Average characters spoken per second

    Returns:
        Estimated duration in seconds
    """
    return len(text) / chars_per_second
