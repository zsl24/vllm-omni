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


# ── OLA crossfade for streaming audio ────────────────────────────────

# ~21 ms at 24 kHz — long enough to smooth spectral discontinuities at
# chunk boundaries while keeping the buffering delay acceptable.
DEFAULT_OVERLAP_SAMPLES = 512

# RMS threshold below which a signal region is treated as silence.
# When the previous chunk's tail is silent but the next chunk starts
# loud (e.g. a consonant onset), a full OLA crossfade would suppress
# the onset.  Skipping the crossfade preserves the signal.
SILENCE_THRESHOLD = 0.01

# Short fade-in length used when skipping OLA at silent→loud boundaries.
# Just enough (~2.7 ms at 24 kHz) to avoid a DC-click without eating
# the consonant onset.
ONSET_FADE_SAMPLES = 64


def _hann_fade_in(n: int) -> np.ndarray:
    """Half-Hann ramp 0→1 of length *n*."""
    if n <= 0:
        return np.array([], dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    return 0.5 * (1 - np.cos(np.pi * t))


def _hann_fade_out(n: int) -> np.ndarray:
    """Half-Hann ramp 1→0 of length *n*."""
    if n <= 0:
        return np.array([], dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    return 0.5 * (1 + np.cos(np.pi * t))


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def ola_crossfade_chunk(
    chunk: np.ndarray,
    is_first_chunk: bool,
    is_last_chunk: bool = False,
    overlap_samples: int = DEFAULT_OVERLAP_SAMPLES,
    prev_tail: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Overlap-add (OLA) crossfade for a streaming audio chunk.

    For the first chunk, applies a fade-in ramp from zero and holds back
    the last ``overlap_samples`` for crossfade with the next chunk.

    For middle chunks, crossfades the held-back tail of the previous
    chunk with the head of the current chunk using Hann windows, then
    holds back the new tail.  When the previous tail is near-silence
    but the current chunk has a strong onset, the crossfade is skipped
    to avoid suppressing the consonant; a short fade-in is applied
    instead.

    For the last chunk, crossfades as above but does *not* hold back the
    tail; instead applies a fade-out to avoid a pop at stream end.

    Args:
        chunk: 1-D float32/float64 audio array for the current chunk.
        is_first_chunk: True if this is the first audio chunk.
        is_last_chunk: True if this is the last audio chunk.
        overlap_samples: Number of samples for overlap/crossfade.
        prev_tail: The tail buffer returned by the previous call.

    Returns:
        (output, new_tail).  *output* contains the samples to send to
        the client.  *new_tail* is the tail buffer to pass as
        ``prev_tail`` on the next call (``None`` for the last chunk).
    """
    chunk = chunk.astype(np.float32, copy=True)
    ov = min(overlap_samples, len(chunk))

    # ── First chunk ──────────────────────────────────────────────
    if is_first_chunk or prev_tail is None:
        chunk[:ov] *= _hann_fade_in(ov)
        if is_last_chunk:
            out_len = min(ov, len(chunk) // 2)
            if out_len > 0:
                chunk[-out_len:] *= _hann_fade_out(out_len)
            return chunk, None
        tail = chunk[-ov:].copy() if len(chunk) > ov else chunk.copy()
        output = chunk[:-ov] if len(chunk) > ov else np.array([], dtype=np.float32)
        return output, tail

    # ── Middle / last chunk ──────────────────────────────────────
    ov = min(ov, len(prev_tail))
    tail_rms = _rms(prev_tail[-ov:])
    head_rms = _rms(chunk[:ov])

    if tail_rms < SILENCE_THRESHOLD and head_rms >= SILENCE_THRESHOLD:
        # Silent→loud: full OLA would suppress the onset.
        # Apply a short fade-in from the tail's last sample instead.
        fade_len = min(ONSET_FADE_SAMPLES, len(chunk))
        dc_offset = float(prev_tail[-1])
        ramp = _hann_fade_in(fade_len)
        chunk[:fade_len] = dc_offset * (1 - ramp) + chunk[:fade_len] * ramp

        if is_last_chunk:
            out_len = min(ov, len(chunk) // 2)
            if out_len > 0:
                chunk[-out_len:] *= _hann_fade_out(out_len)
            return chunk, None

        if len(chunk) <= ov:
            tail = chunk.copy()
            return np.array([], dtype=np.float32), tail

        tail = chunk[-ov:].copy()
        output = chunk[:-ov]
        return output, tail

    # Normal OLA crossfade
    fade_out = _hann_fade_out(ov)
    fade_in = _hann_fade_in(ov)
    crossfaded = prev_tail[-ov:] * fade_out + chunk[:ov] * fade_in

    remainder = chunk[ov:]

    if is_last_chunk:
        out_len = min(ov, len(remainder) // 2)
        if out_len > 0:
            remainder[-out_len:] *= _hann_fade_out(out_len)
        output = np.concatenate([crossfaded, remainder])
        return output, None

    # Hold back tail for next crossfade
    if len(remainder) <= ov:
        tail = remainder.copy()
        return crossfaded, tail

    tail = remainder[-ov:].copy()
    output = np.concatenate([crossfaded, remainder[:-ov]])
    return output, tail


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
