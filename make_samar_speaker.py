"""
Utility script to create a speaker embedding for `aakash.wav`.

Usage (from repo root):
    cd kani-tts-2-openai-server
    python make_aakash_speaker.py

Requirements:
    pip install transformers torchaudio
"""

from pathlib import Path

import torch

from speaker_embedder import compute_speaker_embedding


def main():
    repo_dir = Path(__file__).parent
    voices_dir = repo_dir / "voices"
    speakers_dir = repo_dir / "speakers"
    speakers_dir.mkdir(exist_ok=True)

    audio_path = voices_dir / "samar.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(f"Could not find {audio_path}. Make sure the file exists.")

    print(f"🎙  Computing speaker embedding from {audio_path} ...")
    emb = compute_speaker_embedding(str(audio_path))  # [1, 128]

    # Our server expects .pt files that load to a 1D tensor and then get unsqueezed.
    emb_vec = emb[0].cpu()  # [128]

    out_path = speakers_dir / "samar.pt"
    torch.save(emb_vec, out_path)
    print(f"✅ Saved Aakash speaker embedding to {out_path}")


if __name__ == "__main__":
    main()

