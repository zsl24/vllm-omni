import numpy as np
import pytest

from vllm_omni.utils.audio import (
    DEFAULT_OVERLAP_SAMPLES,
    _hann_fade_in,
    _hann_fade_out,
    ola_crossfade_chunk,
)


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
        assert tail.shape[0] == DEFAULT_OVERLAP_SAMPLES

    def test_first_chunk_hold_back_and_fade_in(self):
        chunk = np.ones(1024, dtype=np.float32)
        output, tail = ola_crossfade_chunk(chunk, is_first_chunk=True)
        # Output should be shorter by overlap_samples
        assert len(output) == len(chunk) - DEFAULT_OVERLAP_SAMPLES
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
        chunk = np.ones(100, dtype=np.float32)
        output, tail = ola_crossfade_chunk(
            chunk, is_first_chunk=True, overlap_samples=512
        )
        # Should handle gracefully without crashing
        assert isinstance(output, np.ndarray)

    def test_default_overlap_samples(self):
        assert DEFAULT_OVERLAP_SAMPLES == 512

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
