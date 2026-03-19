"""
Utility script to create speaker embeddings for all .wav files in voices/.

Usage (from repo root):
    cd kani-tts-2-openai-server
    python make_samar_speaker.py

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

    if not voices_dir.exists():
        raise FileNotFoundError(f"Voices directory not found: {voices_dir}")

    wav_files = sorted(voices_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {voices_dir}")
        return

    print(f"🎙  Creating embeddings for {len(wav_files)} voice(s) in {voices_dir} ...\n")

    for audio_path in wav_files:
        name = audio_path.stem
        out_path = speakers_dir / f"{name}.pt"

        try:
            print(f"  Processing {audio_path.name} ...")
            emb = compute_speaker_embedding(str(audio_path))  # [1, 128]
            emb_vec = emb[0].cpu()  # [128]
            torch.save(emb_vec, out_path)
            print(f"  ✅ Saved {name}.pt")
        except Exception as e:
            print(f"  ⚠️  Failed {audio_path.name}: {e}")

    print(f"\n✅ Done. Embeddings saved to {speakers_dir}")


if __name__ == "__main__":
    main()
