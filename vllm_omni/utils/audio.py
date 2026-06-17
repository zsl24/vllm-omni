# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audio utility functions shared across models and entrypoints."""

import numpy as np
import torch
from torchaudio.functional import melscale_fbanks

from vllm_omni.metrics import definitions as _metric_defs
from vllm_omni.outputs import OmniRequestOutput


def mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """Compute a mel filterbank matrix.

    Drop-in replacement for ``librosa.filters.mel`` using
    ``torchaudio.functional.melscale_fbanks``.

    Args:
        sr: Sample rate of the audio.
        n_fft: FFT window size.
        n_mels: Number of mel bands.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). Defaults to ``sr / 2``.

    Returns:
        Tensor of shape ``(n_mels, n_fft // 2 + 1)``.
    """
    if fmax is None:
        fmax = float(sr) / 2.0
    # Use mel_scale='slaney' and norm='slaney' to match librosa's
    # default behaviour (Slaney 1998 frequency mapping with area
    # normalization).
    return melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=float(fmin),
        f_max=float(fmax),
        n_mels=n_mels,
        sample_rate=sr,
        mel_scale="slaney",
        norm="slaney",
    ).T


def peak_normalize(
    audio: np.ndarray,
    db_level: float = -6.0,
) -> np.ndarray:
    """Normalize audio so peak amplitude reaches a target dB level.

    Drop-in replacement for ``sox.Transformer().norm(db_level=...)``.

    Args:
        audio: Input waveform as a 1-D numpy array.
        db_level: Target peak amplitude in dBFS.

    Returns:
        Normalized waveform with the same dtype as *audio*.
    """
    peak = np.abs(audio).max()
    if peak == 0:
        return audio
    target = 10.0 ** (db_level / 20.0)
    return audio * (target / peak)


def _hann_fade_in(n: int) -> np.ndarray:
    """Generate a Hann fade-in curve of length n (0→1)."""
    if n <= 0:
        return np.array([], dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    return 0.5 * (1 - np.cos(np.pi * t))


def _hann_fade_out(n: int) -> np.ndarray:
    """Generate a Hann fade-out curve of length n (1→0)."""
    if n <= 0:
        return np.array([], dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    return 0.5 * (1 + np.cos(np.pi * t))


# ~3ms at 24kHz, enough to smooth boundary discontinuities without
# creating audible volume dips.  Set to 0 to disable.
DEFAULT_BLEND_SAMPLES = 64


def apply_hann_fade_to_chunk(
    chunk: np.ndarray,
    is_first_chunk: bool,
    is_last_chunk: bool = False,
    blend_samples: int = DEFAULT_BLEND_SAMPLES,
) -> np.ndarray:
    """Apply Hann-window fade-in/fade-out to a streaming audio chunk.

    For vllm-omni's non-overlapping contiguous chunks, this uses a pure
    fade approach (NOT crossfade blending) to smooth boundary
    discontinuities without losing or duplicating frames.

    Qwen3-TTS uses crossfade+trim because its sliding-window re-decode
    produces overlapping chunks where the trim removes redundant samples.
    vllm-omni's chunked_decode_streaming produces non-overlapping chunks,
    so trim would discard real audio content → "swallowed audio".

    Every chunk receives fade-in at the head (avoids pop at audio start
    on the first chunk; smooths boundary discontinuity on subsequent
    chunks) and fade-out at the tail (smooths boundary on non-last
    chunks; avoids pop on audio completion on the last chunk).

    Args:
        chunk: 1-D float32 audio array for the current chunk.
        is_first_chunk: True if this is the first audio chunk.
        is_last_chunk: True if this is the last audio chunk.
        blend_samples: Number of samples for fade length.

    Returns:
        The chunk after fade processing (new array).
    """
    chunk = chunk.copy()
    # Cap fade length to half the chunk to prevent overlap between
    # the fade-in and fade-out regions on short chunks.
    fade_len = min(blend_samples, len(chunk) // 2)
    if fade_len <= 0:
        return chunk

    chunk[:fade_len] *= _hann_fade_in(fade_len)
    chunk[-fade_len:] *= _hann_fade_out(fade_len)

    return chunk


def audio_chunk_pcm_bytes(omni_res: OmniRequestOutput) -> int:
    """Best-effort PCM byte count of the last audio chunk in ``omni_res``.

    Used by the audio-streaming continuity tracker to size the player buffer.
    Returns 0 when the chunk shape can't be interpreted — caller drops the
    sample rather than recording a wrong byte count.
    """
    try:
        final_res = omni_res.request_output
        mm_output = final_res.outputs[0].multimodal_output
        audio_data = mm_output.get("audio")
        if isinstance(audio_data, list):
            if not audio_data:
                return 0
            audio_tensor = audio_data[-1]
        else:
            audio_tensor = audio_data
        if audio_tensor is None:
            return 0
        n_samples = int(audio_tensor.numel() if hasattr(audio_tensor, "numel") else audio_tensor.size)
        # PCM s16le mono → 2 bytes per sample. This matches what
        # _create_audio_choice serialises via CreateAudio (response_format="wav").
        return max(n_samples, 0) * 2
    except Exception:
        return 0


def audio_chunk_sample_rate(omni_res: OmniRequestOutput) -> int:
    """Resolve audio sample rate for the request's audio stream."""
    try:
        mm_output = omni_res.request_output.outputs[0].multimodal_output
    except Exception:
        return _metric_defs.DEFAULT_AUDIO_SAMPLE_RATE
    return _metric_defs.resolve_audio_sample_rate(mm_output)
