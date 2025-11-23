"""Speaker verification helper for gating recognition by target voice."""

from __future__ import annotations

import contextlib
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


@dataclass
class VerificationResult:
    score: float
    accepted: bool


class SpeakerVerifier:
    """Compute speaker embeddings and score similarity against a target profile."""

    def __init__(
        self,
        reference_path: str,
        *,
        device: str = "cuda",
        threshold: float = 0.75,
    ) -> None:
        self._target_sr = 16000
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self._feature = T.MFCC(
            sample_rate=self._target_sr,
            n_mfcc=40,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 64,
                "f_min": 20,
                "f_max": 7600,
            },
        ).to(self._device)
        self._threshold = threshold
        self._ref_embedding = self._compute_reference(reference_path)

    def verify(self, audio: torch.Tensor, sample_rate: int) -> VerificationResult:
        """Return similarity score and acceptance flag for the provided audio segment."""
        embedding = self._embed(self._prepare(audio, sample_rate))
        score = torch.nn.functional.cosine_similarity(embedding, self._ref_embedding).item()
        return VerificationResult(score=score, accepted=score >= self._threshold)

    def _compute_reference(self, path: str) -> torch.Tensor:
        waveform, sample_rate = _load_audio(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        prepared = self._prepare(waveform, sample_rate)
        return self._embed(prepared)

    def _prepare(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self._target_sr:
            waveform = F.resample(waveform, sample_rate, self._target_sr)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.to(self._device)

    def _embed(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mfcc = self._feature(waveform)
        stats = torch.cat([mfcc.mean(dim=-1), mfcc.std(dim=-1)], dim=-1)
        embedding = torch.nn.functional.normalize(stats, dim=1)
        return embedding.cpu()


def create_speaker_verifier(
    reference_path: Optional[str],
    *,
    device: str = "cuda",
    threshold: float = 0.75,
) -> Optional[SpeakerVerifier]:
    if not reference_path:
        return None
    return SpeakerVerifier(reference_path, device=device, threshold=threshold)


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(path)
    except RuntimeError:
        if path.lower().endswith(".wav"):
            return _load_wav(path)
        raise


def _load_wav(path: str) -> tuple[torch.Tensor, int]:
    with contextlib.closing(wave.open(path, "rb")) as wf:
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise RuntimeError("仅支持 16-bit PCM WAV 作为声纹参考音频。")
        num_channels = wf.getnchannels()
        num_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        pcm_bytes = wf.readframes(num_frames)

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return waveform, sample_rate
