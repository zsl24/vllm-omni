import numpy as np
import pytest

from vllm_omni.utils.audio import (
    DEFAULT_OVERLAP_SECONDS,
    DEFAULT_ONSET_FADE_SECONDS,
    SILENCE_THRESHOLD,
    _hann_fade_in,
    _hann_fade_out,
    ola_crossfade_chunk,
)


def _overlap_samples(sr: int = 24000) -> int:
    return max(int(DEFAULT_OVERLAP_SECONDS * sr), 1)


def _onset_fade_samples(sr: int = 24000) -> int:
    return max(int(DEFAULT_ONSET_FADE_SECONDS * sr), 1)


class TestHannFadeCurves:
    def test_fade_in_shape_and_bounds(self):
        n = 64
        curve = _hann_fade_in(n)
        assert curve.shape == (n,)
        assert curve.dtype == np.float32
        assert np.isclose(curve[0], 0.0, atol=1e-6)
        assert np.isclose(curve[-1], 1.0, atol=1e-6)
        assert np.all(curve >= 0) and np.all(curve <= 1)

    def test_fade_out_shape_and_bounds(self):
        n = 64
        curve = _hann_fade_out(n)
        assert curve.shape == (n,)
        assert curve.dtype == np.float32
        assert np.isclose(curve[0], 1.0, atol=1e-6)
        assert np.isclose(curve[-1], 0.0, atol=1e-6)
        assert np.all(curve >= 0) and np.all(curve <= 1)

    def test_fade_in_out_sum_to_one(self):
        n = 32
        fade_in = _hann_fade_in(n)
        fade_out = _hann_fade_out(n)
        np.testing.assert_allclose(fade_in + fade_out, 1.0, atol=1e-6)

    def test_zero_length_returns_empty(self):
        assert _hann_fade_in(0).size == 0
        assert _hann_fade_out(0).size == 0

    def test_negative_returns_empty(self):
        assert _hann_fade_in(-1).size == 0
        assert _hann_fade_out(-1).size == 0


class TestOlaCrossfadeChunk:
    def test_first_chunk_returns_tail(self):
        chunk = np.ones(1024, dtype=np.float32)
        output, tail = ola_crossfade_chunk(chunk, is_first_chunk=True)
        assert output.dtype == np.float32
        assert tail is not None
        assert tail.shape[0] == _overlap_samples()

    def test_first_chunk_hold_back_and_fade_in(self):
        chunk = np.ones(1024, dtype=np.float32)
        output, tail = ola_crossfade_chunk(chunk, is_first_chunk=True)
        # Output should be shorter by overlap_samples
        assert len(output) == len(chunk) - _overlap_samples()
        # Head should be faded in (near zero at start)
        assert output[0] < 0.1
        # Body should be untouched
        assert np.isclose(output[-1], 1.0)

    def test_first_and_last_chunk_no_tail(self):
        chunk = np.ones(1024, dtype=np.float32)
        output, tail = ola_crossfade_chunk(
            chunk, is_first_chunk=True, is_last_chunk=True
        )
        assert tail is None
        # Should have fade-in and fade-out
        assert output[0] < 0.1
        assert output[-1] < 0.1

    def test_middle_chunk_crossfade(self):
        prev = np.full(1024, 0.3, dtype=np.float32)
        curr = np.full(1024, 0.7, dtype=np.float32)
        _, tail = ola_crossfade_chunk(prev, is_first_chunk=True)
        output, new_tail = ola_crossfade_chunk(
            curr, is_first_chunk=False, prev_tail=tail
        )
        # Output should start at prev's tail amplitude (crossfade)
        assert output[0] < 0.7  # blended, not starting at 0.7
        # After crossfade, signal should reach curr's amplitude
        assert np.isclose(output[-1], 0.7, atol=0.05)

    def test_boundary_continuity(self):
        """Output should be amplitude-continuous at chunk boundaries."""
        chunks = [
            np.full(1024, a, dtype=np.float32)
            for a in [0.2, 0.5, 0.8, 0.3, 0.6]
        ]
        all_output = []
        tail = None
        for i, chunk in enumerate(chunks):
            is_first = i == 0
            is_last = i == len(chunks) - 1
            output, tail = ola_crossfade_chunk(
                chunk,
                is_first_chunk=is_first,
                is_last_chunk=is_last,
                prev_tail=tail,
            )
            all_output.append(output)

        merged = np.concatenate(all_output)
        # Check no large jumps at the concatenation points
        for i in range(1, len(merged)):
            jump = abs(merged[i] - merged[i - 1])
            assert jump < 0.05, f"Jump of {jump:.4f} at sample {i}"

    def test_energy_conservation_in_crossfade(self):
        """Hann windows sum to 1, so crossfade preserves energy."""
        n = 512
        fade_out = _hann_fade_out(n)
        fade_in = _hann_fade_in(n)
        np.testing.assert_allclose(fade_out + fade_in, 1.0, atol=1e-6)

    def test_empty_chunk(self):
        chunk = np.array([], dtype=np.float32)
        output, tail = ola_crossfade_chunk(chunk, is_first_chunk=True)
        assert output.size == 0

    def test_chunk_shorter_than_overlap(self):
        # With 24kHz sample rate, overlap = 512 samples.
        # A 100-sample chunk is shorter than overlap.
        chunk = np.ones(100, dtype=np.float32)
        output, tail = ola_crossfade_chunk(chunk, is_first_chunk=True)
        # Should handle gracefully without crashing
        assert isinstance(output, np.ndarray)

    def test_default_overlap_duration(self):
        assert np.isclose(DEFAULT_OVERLAP_SECONDS, 0.0213)

    def test_multi_chunk_chain_no_gaps(self):
        """Full chain should produce seamless audio."""
        np.random.seed(42)
        chunks = [np.random.randn(2048).astype(np.float32) * 0.3 for _ in range(10)]

        all_output = []
        tail = None
        for i, chunk in enumerate(chunks):
            output, tail = ola_crossfade_chunk(
                chunk,
                is_first_chunk=(i == 0),
                is_last_chunk=(i == len(chunks) - 1),
                prev_tail=tail,
            )
            all_output.append(output)

        merged = np.concatenate(all_output)
        # No NaN or Inf
        assert np.all(np.isfinite(merged))
        # Reasonable amplitude range
        assert np.max(np.abs(merged)) < 10.0


class TestSampleRateAdaptation:
    """Verify that OLA overlap and onset fade scale with sample rate."""

    @pytest.mark.parametrize("sr", [16000, 22050, 24000, 44100])
    def test_overlap_scales_with_sample_rate(self, sr):
        overlap = _overlap_samples(sr)
        onset_fade = _onset_fade_samples(sr)
        # Overlap duration should be ~21.3 ms at any sample rate
        overlap_ms = overlap / sr * 1000
        assert 18 < overlap_ms < 25, f"sr={sr}: overlap={overlap_ms:.1f}ms"
        # Onset fade duration should be ~2.7 ms at any sample rate
        onset_ms = onset_fade / sr * 1000
        assert 1.5 < onset_ms < 4.0, f"sr={sr}: onset_fade={onset_ms:.1f}ms"

    def test_16khz_has_fewer_overlap_samples_than_24khz(self):
        """Same duration → lower sample rate needs fewer samples in the overlap window."""
        assert _overlap_samples(16000) < _overlap_samples(24000)

    def test_44khz_has_more_overlap_samples_than_24khz(self):
        """Same duration → higher sample rate needs more samples in the overlap window."""
        assert _overlap_samples(44100) > _overlap_samples(24000)

    @pytest.mark.parametrize("sr", [16000, 24000, 44100])
    def test_first_chunk_tail_size_matches_sample_rate(self, sr):
        """Tail length should equal overlap_samples for the given sample rate."""
        chunk = np.ones(2048, dtype=np.float32)
        _, tail = ola_crossfade_chunk(chunk, is_first_chunk=True, sample_rate=sr)
        assert tail is not None
        assert tail.shape[0] == _overlap_samples(sr)

    @pytest.mark.parametrize("sr", [16000, 24000, 44100])
    def test_crossfade_works_at_various_sample_rates(self, sr):
        """End-to-end OLA chain should produce finite audio at any sample rate."""
        chunks = [np.random.randn(2048).astype(np.float32) * 0.3 for _ in range(4)]
        all_output = []
        tail = None
        for i, chunk in enumerate(chunks):
            output, tail = ola_crossfade_chunk(
                chunk,
                is_first_chunk=(i == 0),
                is_last_chunk=(i == len(chunks) - 1),
                sample_rate=sr,
                prev_tail=tail,
            )
            all_output.append(output)
        merged = np.concatenate(all_output)
        assert np.all(np.isfinite(merged))


class TestEnergyAdaptiveOla:
    """Tests for the silent→loud onset-preservation logic."""

    def test_silent_to_loud_preserves_onset(self):
        """When prev tail is silent and next chunk is loud, onset recovers within ~64 samples.

        A 64-sample fade-in (~2.7ms at 24kHz) is used instead of the full 512-sample
        crossfade (~21ms), so the onset is preserved much better than plain OLA.
        Sample 0 may still be near 0 when DC offset is 0, but the signal reaches
        full amplitude quickly.
        """
        silence = np.zeros(512, dtype=np.float32)
        loud = np.full(1024, 0.8, dtype=np.float32)

        _, tail = ola_crossfade_chunk(silence, is_first_chunk=True)
        output, _ = ola_crossfade_chunk(loud, is_first_chunk=False, prev_tail=tail)

        # With short fade-in (64 samples), signal should be near full amplitude
        # by sample 80 (a few samples past the fade window).
        assert output[80] > 0.7, f"Onset not recovered: output[80]={output[80]:.4f}"

    def test_silent_to_loud_faster_than_normal_ola(self):
        """Energy-adaptive fade-in should recover faster than normal OLA."""
        silence = np.zeros(512, dtype=np.float32)
        loud = np.full(1024, 0.8, dtype=np.float32)

        # With energy-adaptive OLA
        _, tail = ola_crossfade_chunk(silence, is_first_chunk=True)
        output_adaptive, _ = ola_crossfade_chunk(
            loud.copy(), is_first_chunk=False, prev_tail=tail
        )

        # With normal OLA (force both sides above threshold)
        loudish_tail = np.full(512, 0.05, dtype=np.float32)  # above threshold
        _, normal_tail = ola_crossfade_chunk(
            np.full(1024, 0.05, dtype=np.float32), is_first_chunk=True
        )
        # Replace tail with a signal that triggers normal OLA
        normal_tail = np.full(512, 0.3, dtype=np.float32)
        output_normal, _ = ola_crossfade_chunk(
            loud.copy(), is_first_chunk=False, prev_tail=normal_tail
        )

        # At sample 64, adaptive output should be louder (faster recovery)
        assert output_adaptive[64] >= output_normal[64]

    def test_silent_to_loud_short_fade_in(self):
        """The short fade-in should smoothly transition from DC offset to signal."""
        dc_val = 0.02
        silence = np.full(512, dc_val, dtype=np.float32) * 0.01  # near-silent
        loud = np.full(1024, 0.9, dtype=np.float32)

        _, tail = ola_crossfade_chunk(silence, is_first_chunk=True)
        output, _ = ola_crossfade_chunk(loud, is_first_chunk=False, prev_tail=tail)

        # After the short fade-in region (64 samples), signal should be at full amplitude
        assert np.isclose(output[100], 0.9, atol=0.05)

    def test_loud_to_loud_uses_normal_ola(self):
        """When both sides have signal, normal OLA crossfade is used."""
        prev_chunk = np.full(1024, 0.5, dtype=np.float32)
        curr_chunk = np.full(1024, 0.7, dtype=np.float32)

        _, tail = ola_crossfade_chunk(prev_chunk, is_first_chunk=True)
        output, _ = ola_crossfade_chunk(curr_chunk, is_first_chunk=False, prev_tail=tail)

        # Crossfade blends both signals; first sample should be between 0.5 and 0.7
        assert 0.3 < output[0] < 0.7

    def test_silent_to_silent_uses_normal_ola(self):
        """When both sides are near-silent, normal OLA is fine (doesn't matter)."""
        prev_chunk = np.zeros(1024, dtype=np.float32)
        curr_chunk = np.zeros(1024, dtype=np.float32)

        _, tail = ola_crossfade_chunk(prev_chunk, is_first_chunk=True)
        output, new_tail = ola_crossfade_chunk(
            curr_chunk, is_first_chunk=False, prev_tail=tail
        )

        # Should still work, output is near-zero
        assert np.max(np.abs(output)) < 0.01

    def test_loud_to_silent_uses_normal_ola(self):
        """When prev is loud and next is silent, normal OLA is fine."""
        prev_chunk = np.full(1024, 0.8, dtype=np.float32)
        curr_chunk = np.zeros(1024, dtype=np.float32)

        _, tail = ola_crossfade_chunk(prev_chunk, is_first_chunk=True)
        output, _ = ola_crossfade_chunk(curr_chunk, is_first_chunk=False, prev_tail=tail)

        # Crossfade from loud to silent — should blend smoothly
        assert np.all(np.isfinite(output))

    def test_silent_to_loud_last_chunk(self):
        """Silent→loud with is_last_chunk=True should apply fade-out at end."""
        dc_val = 0.01
        silence = np.full(512, dc_val, dtype=np.float32) * 0.01
        loud = np.full(1024, 0.8, dtype=np.float32)

        _, tail = ola_crossfade_chunk(silence, is_first_chunk=True)
        output, new_tail = ola_crossfade_chunk(
            loud, is_first_chunk=False, is_last_chunk=True, prev_tail=tail
        )

        assert new_tail is None  # last chunk returns no tail
        assert output[80] > 0.5  # onset preserved after short fade-in
        assert output[-1] < 0.5  # fade-out applied at end

    @pytest.mark.parametrize("sr", [16000, 24000, 44100])
    def test_silent_to_loud_preserves_onset_at_any_sample_rate(self, sr):
        """Onset preservation should work at any sample rate."""
        onset_fade = _onset_fade_samples(sr)
        silence = np.zeros(_overlap_samples(sr), dtype=np.float32)
        loud = np.full(2048, 0.8, dtype=np.float32)

        _, tail = ola_crossfade_chunk(silence, is_first_chunk=True, sample_rate=sr)
        output, _ = ola_crossfade_chunk(
            loud, is_first_chunk=False, sample_rate=sr, prev_tail=tail
        )

        # A few samples past the onset fade window, signal should be near full
        check_idx = min(onset_fade + 16, len(output) - 1)
        assert output[check_idx] > 0.7, (
            f"sr={sr}: onset not recovered at idx {check_idx}, val={output[check_idx]:.4f}"
        )
