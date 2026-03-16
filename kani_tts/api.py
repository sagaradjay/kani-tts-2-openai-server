"""Simple API for Kani-TTS."""
from typing import Tuple, Optional, Union, Dict
from pathlib import Path
import numpy as np
import logging
import warnings
import torch
import time
from .core import TTSConfig, NemoAudioPlayer, KaniModel


def suppress_all_logs():
    """
    Suppress all logging output from transformers, NeMo, PyTorch, and other libraries.
    Only print() statements from user code will be visible.
    """
    # Suppress Python warnings
    warnings.filterwarnings('ignore')

    # Suppress transformers logs
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except ImportError:
        pass

    # Suppress NeMo logs
    logging.getLogger('nemo').setLevel(logging.ERROR)
    logging.getLogger('nemo_logger').setLevel(logging.ERROR)

    # Suppress PyTorch logs
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('pytorch').setLevel(logging.ERROR)

    # Suppress other common loggers
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)

    # Set root logger to ERROR level
    logging.getLogger().setLevel(logging.ERROR)


class KaniTTS:
    """
    Simple interface for Kani text-to-speech model.

    Example:
        >>> model = KaniTTS('your-model-name')
        >>> audio, text = model("Hello, world!")
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        max_new_tokens: int = 1200,
        temperature: float = 1,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        tokeniser_length: int = 64400,
        suppress_logs: bool = True,
        show_info: bool = True,
        text_vocab_size: int = 64400,
        tokens_per_frame: int = 4,
        audio_step: float = 0.5,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        speaker_emb_dim: int = 128,
        track_rtf: bool = False,
        attn_implementation: str = "sdpa",
        use_cuda_graphs: bool = False,
    ):
        """
        Initialize Kani-TTS model.

        Args:
            model_name: Hugging Face model ID or path to local model
            device_map: Device mapping for model (default: "auto")
            max_new_tokens: Maximum number of tokens to generate (default: 1800)
            temperature: Sampling temperature (default: 0.6)
            top_p: Top-p sampling parameter (default: 0.95)
            repetition_penalty: Repetition penalty (default: 1.1)
            tokeniser_length: Length of text tokenizer vocabulary (default: 64400)
            suppress_logs: Whether to suppress library logs (default: True)
            show_info: Whether to display model info on initialization (default: True)
            text_vocab_size: Text vocabulary size for position encoding (default: 64400)
            tokens_per_frame: Number of audio tokens per frame (default: 4)
            audio_step: Position step size per audio frame (default: 0.5)
            use_learnable_rope: Enable learnable RoPE with per-layer alpha (default: False)
            alpha_min: Minimum alpha value for learnable RoPE (default: 0.1)
            alpha_max: Maximum alpha value for learnable RoPE (default: 2.0)
            speaker_emb_dim: Dimension of speaker embeddings (default: 128)
            track_rtf: Enable Real Time Factor tracking (default: False)
            attn_implementation: Attention implementation ("sdpa", "flash_attention_2", "eager")
            use_cuda_graphs: Enable CUDA graphs for decode phase (default: False, 1.3-1.5x speedup)
        """
        if suppress_logs:
            suppress_all_logs()

        self.config = TTSConfig(
            device_map=device_map,
            tokeniser_length=tokeniser_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            text_vocab_size=text_vocab_size,
            tokens_per_frame=tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            speaker_emb_dim=speaker_emb_dim,
            attn_implementation=attn_implementation,
            use_cuda_graphs=use_cuda_graphs,
        )
        self.model_name = model_name

        self.player = NemoAudioPlayer(self.config)
        self.model = KaniModel(self.config, model_name, self.player)
        self.status = self.model.status
        self.speaker_list = self.model.speaker_list
        self.sample_rate = self.config.sample_rate

        # RTF tracking
        self.track_rtf = track_rtf
        self.last_rtf_metrics: Optional[Dict[str, float]] = None

        if show_info:
            self.show_model_info()

    def __call__(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        speaker_emb: Optional[Union[torch.Tensor, str, Path]] = None,
        return_rtf: bool = False
    ) -> Union[Tuple[np.ndarray, str], Tuple[np.ndarray, str, Dict[str, float]]]:
        """
        Generate audio from text.

        Args:
            text: Input text to convert to speech
            speaker_id: Optional speaker ID for multi-speaker models
            speaker_emb: Optional speaker embedding. Can be:
                - torch.Tensor: [1, speaker_emb_dim] or [speaker_emb_dim]
                - str/Path: Path to .pt file containing speaker embedding
            return_rtf: If True, return RTF metrics as third element (default: False)

        Returns:
            If return_rtf is False:
                Tuple of (audio_waveform, text)
            If return_rtf is True:
                Tuple of (audio_waveform, text, rtf_metrics)
            where rtf_metrics is a dict with 'generation_time', 'audio_duration', 'rtf'
        """
        return self.generate(text, speaker_id, speaker_emb, return_rtf=return_rtf)

    def generate(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        speaker_emb: Optional[Union[torch.Tensor, str, Path]] = None,
        return_rtf: bool = False
    ) -> Union[Tuple[np.ndarray, str], Tuple[np.ndarray, str, Dict[str, float]]]:
        """
        Generate audio from text.

        Args:
            text: Input text to convert to speech
            speaker_id: Optional speaker ID for multi-speaker models
            speaker_emb: Optional speaker embedding. Can be:
                - torch.Tensor: [1, speaker_emb_dim] or [speaker_emb_dim]
                - str/Path: Path to .pt file containing speaker embedding
            return_rtf: If True, return RTF metrics as third element (default: False)

        Returns:
            If return_rtf is False:
                Tuple of (audio_waveform, text)
            If return_rtf is True:
                Tuple of (audio_waveform, text, rtf_metrics)
            where rtf_metrics is a dict with 'generation_time', 'audio_duration', 'rtf'
        """
        # Load speaker embedding if path is provided
        if speaker_emb is not None and not isinstance(speaker_emb, torch.Tensor):
            speaker_emb = self.load_speaker_embedding(speaker_emb)

        # Ensure speaker_emb has batch dimension
        if speaker_emb is not None and speaker_emb.ndim == 1:
            speaker_emb = speaker_emb.unsqueeze(0)

        # Track RTF if enabled
        should_track = self.track_rtf or return_rtf

        if should_track:
            start_time = time.time()

        # Generate audio
        audio, text_out = self.model.run_model(text, speaker_id, speaker_emb)

        if should_track:
            # Calculate metrics
            generation_time = time.time() - start_time
            audio_duration = len(audio) / self.sample_rate
            rtf = generation_time / audio_duration if audio_duration > 0 else 0

            # Store metrics
            self.last_rtf_metrics = {
                'generation_time': generation_time,
                'audio_duration': audio_duration,
                'rtf': rtf
            }

            # Print if tracking is enabled globally
            if self.track_rtf:
                print(f"\n{'='*60}")
                print(f"⏱️  Generation Time: {generation_time:.3f}s")
                print(f"🎵 Audio Duration: {audio_duration:.3f}s")
                print(f"🚀 RTF: {rtf:.3f}x ({'faster' if rtf < 1.0 else 'slower'} than real-time)")
                print(f"{'='*60}\n")

            # Return with RTF if requested
            if return_rtf:
                return audio, text_out, self.last_rtf_metrics

        return audio, text_out

    def load_speaker_embedding(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load speaker embedding from a .pt file.

        Args:
            path: Path to .pt file containing speaker embedding

        Returns:
            Speaker embedding tensor [speaker_emb_dim] or [1, speaker_emb_dim]
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Speaker embedding file not found: {path}")

        if path.suffix != '.pt':
            raise ValueError(f"Speaker embedding must be a .pt file, got: {path.suffix}")

        speaker_emb = torch.load(path)

        # Validate shape
        if speaker_emb.ndim == 1:
            expected_dim = self.config.speaker_emb_dim
            if speaker_emb.shape[0] != expected_dim:
                raise ValueError(
                    f"Speaker embedding has wrong dimension: expected {expected_dim}, "
                    f"got {speaker_emb.shape[0]}"
                )
        elif speaker_emb.ndim == 2:
            if speaker_emb.shape[1] != self.config.speaker_emb_dim:
                raise ValueError(
                    f"Speaker embedding has wrong dimension: expected [..., {self.config.speaker_emb_dim}], "
                    f"got {speaker_emb.shape}"
                )
        else:
            raise ValueError(f"Speaker embedding must be 1D or 2D, got shape: {speaker_emb.shape}")

        return speaker_emb

    def get_rtf_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get the RTF metrics from the last generation.

        Returns:
            Dictionary with 'generation_time', 'audio_duration', 'rtf' keys,
            or None if no generation has been performed yet.
        """
        return self.last_rtf_metrics

    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio waveform to file.

        Args:
            audio: Audio waveform as numpy array
            output_path: Path to save audio file (e.g., "output.wav")
        """
        try:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
        except ImportError:
            raise ImportError(
                "soundfile is required to save audio. "
                "Install it with: pip install soundfile"
            )

    def show_model_info(self):
        """
        Display beautiful model information banner.
        """
        print()
        print("╔════════════════════════════════════════════════════════════╗")
        print("║                                                            ║")
        print("║                   N I N E N I N E S I X  😼                ║")
        print("║                                                            ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print()
        print("              /\\_/\\  ")
        print("             ( o.o )")
        print("              > ^ <")
        print()
        print("─" * 62)

        # Model name
        model_display = self.model_name
        if len(model_display) > 50:
            model_display = "..." + model_display[-47:]
        print(f"  Model: {model_display}")

        # Device info
        import torch
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"  Device: {device}")

        # Speaker info
        if self.status == 'multispeaker':
            print(f"  Mode: Multi-speaker ({len(self.speaker_list)} speakers)")
            if self.speaker_list and len(self.speaker_list) <= 5:
                speakers_str = ", ".join(self.speaker_list)
                print(f"  Speakers: {speakers_str}")
            elif self.speaker_list:
                print(f"  Speakers: {self.speaker_list[0]}, {self.speaker_list[1]}, ... (use .show_speakers() to see all)")
        else:
            print(f"  Mode: Single-speaker")

        print()
        print("  Configuration:")
        print(f"    • Sample Rate: {self.sample_rate} Hz")
        print(f"    • Temperature: {self.config.temperature}")
        print(f"    • Top-p: {self.config.top_p}")
        print(f"    • Max Tokens: {self.config.max_new_tokens}")
        print(f"    • Repetition Penalty: {self.config.repetition_penalty}")
        print(f"    • Speaker Embedding Dim: {self.config.speaker_emb_dim}")
        print(f"    • RTF Tracking: {'Enabled' if self.track_rtf else 'Disabled'}")
        print(f"    • BemaTTS: Enabled (frame-level position encoding)")
        print(f"    • Text Vocab Size: {self.config.text_vocab_size}")
        print(f"    • Tokens per Frame: {self.config.tokens_per_frame}")
        print(f"    • Audio Step: {self.config.audio_step}")
        if self.config.use_learnable_rope:
            print(f"    • Learnable RoPE: Enabled (per-layer frequency scaling)")
            print(f"    • Alpha Range: [{self.config.alpha_min}, {self.config.alpha_max}]")
        else:
            print(f"    • Learnable RoPE: Disabled (standard RoPE)")

        print("─" * 62)
        print()
        print("  Ready to generate speech! 🎵")
        print()

    def show_speakers(self):
        """
        Display available speakers for multi-speaker models.

        For single-speaker models, displays a message that speaker selection
        is not available.
        """
        print("=" * 50)
        if self.status == 'multispeaker':
            print("Available Speakers:")
            print("-" * 50)
            if self.speaker_list:
                for i, speaker in enumerate(self.speaker_list, 1):
                    print(f"  {i}. {speaker}")
            else:
                print("  No speakers configured")
        else:
            print("Single-speaker model")
            print("Speaker selection is not available for this model")
        print("=" * 50)
