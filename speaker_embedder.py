"""
Speaker embedder for the OpenAI-compatible server.

Uses SpeechBrain ECAPA-TDNN embeddings so inference matches the rebuilt
training dataset speaker space.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier


class SpeakerEmbedder:
    """
    Generate ECAPA speaker embeddings from raw audio or files.

    Returns 192-dim L2-normalized embeddings compatible with the retrained TTS model.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
        max_duration_sec: float = 30.0,
    ):
        self.model_name = model_name
        self.target_sr = 16000
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(max_duration_sec * self.target_sr)
        self.emb_dim = 192

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🔊 Loading SpeechBrain speaker embedder from {model_name}...")
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": str(self.device)},
        )
        print(f"✅ Speaker embedder ready on {self.device}")

    def _prepare_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if audio.dim() == 2:
            if audio.shape[0] < audio.shape[1]:
                audio = audio.mean(dim=0)
            else:
                audio = audio[0]

        if audio.dim() != 1:
            raise ValueError(f"Expected 1D or 2D audio, got shape {audio.shape}")

        if sample_rate != self.target_sr:
            try:
                import torchaudio.transforms as T
            except ImportError:
                raise ImportError("torchaudio is required for resampling. Install with: pip install torchaudio")

            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            audio = resampler(audio.unsqueeze(0)).squeeze(0)

        if audio.shape[0] == 0:
            raise ValueError("Audio is empty")

        if audio.shape[0] > self.max_samples:
            print(f"⚠️  Audio is {audio.shape[0] / self.target_sr:.2f}s, truncating to {self.max_duration_sec}s")
            audio = audio[:self.max_samples]

        return audio

    def embed_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        else:
            audio = audio.float()

        if sample_rate is None:
            sample_rate = self.target_sr

        audio = self._prepare_audio(audio, sample_rate)
        audio_batch = audio.unsqueeze(0).to(self.device)
        lengths = torch.ones(1, device=self.device)

        with torch.no_grad():
            embedding = self.classifier.encode_batch(audio_batch, wav_lens=lengths)
            embedding = embedding.squeeze(1) if embedding.ndim == 3 else embedding
            embedding = F.normalize(embedding, dim=-1)

        return embedding

    def embed_audio_file(self, audio_path: str) -> torch.Tensor:
        try:
            import torchaudio
        except ImportError:
            raise ImportError("torchaudio is required for loading audio files. Install with: pip install torchaudio")

        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio[0]
        return self.embed_audio(audio, sample_rate=sr)


def compute_speaker_embedding(
    audio: Union[np.ndarray, torch.Tensor, str],
    sample_rate: int = 16000,
    model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
    device: Optional[str] = None,
) -> torch.Tensor:
    embedder = SpeakerEmbedder(model_name=model_name, device=device)
    if isinstance(audio, str):
        return embedder.embed_audio_file(audio)
    return embedder.embed_audio(audio, sample_rate=sample_rate)
