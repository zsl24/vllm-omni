import numpy as np
import pytest

from vllm_omni.utils.audio import (
    DEFAULT_BLEND_SAMPLES,
    _hann_fade_in,
    _hann_fade_out,
    apply_hann_fade_to_chunk,
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

    def test_fade_in_out_are_inverses(self):
        n = 32
        fade_in = _hann_fade_in(n)
        fade_out = _hann_fade_out(n)
        # fade_in + fade_out should sum to 1 (complementary)
        np.testing.assert_allclose(fade_in + fade_out, 1.0, atol=1e-6)

    def test_zero_length_returns_empty(self):
        assert _hann_fade_in(0).size == 0
        assert _hann_fade_out(0).size == 0

    def test_negative_returns_empty(self):
        assert _hann_fade_in(-1).size == 0
        assert _hann_fade_out(-1).size == 0


class TestApplyHannFadeToChunk:
    def test_returns_new_array(self):
        chunk = np.ones(256, dtype=np.float32)
        result = apply_hann_fade_to_chunk(chunk, is_first_chunk=True)
        assert result is not chunk
        assert chunk.dtype == result.dtype

    def test_first_chunk_fade_in_from_zero(self):
        chunk = np.ones(256, dtype=np.float32)
        result = apply_hann_fade_to_chunk(chunk, is_first_chunk=True)
        # Head should ramp up from near-zero
        assert result[0] < 0.1
        # Tail should ramp down to near-zero (every chunk gets fade-out)
        assert result[-1] < 0.1

    def test_middle_chunk_boundary_fade(self):
        chunk = np.ones(256, dtype=np.float32)
        result = apply_hann_fade_to_chunk(chunk, is_first_chunk=False)
        # Head and tail both faded
        assert result[0] < 0.1
        assert result[-1] < 0.1
        # Center should be untouched (= 1.0)
        center = len(chunk) // 2
        assert np.isclose(result[center], 1.0, atol=0.05)

    def test_short_chunk_no_overlap(self):
        # Chunk shorter than 2*blend_samples — fade regions must not overlap
        chunk = np.ones(30, dtype=np.float32)
        result = apply_hann_fade_to_chunk(chunk, is_first_chunk=True, blend_samples=64)
        # fade_len capped to len(chunk)//2 = 15
        # center sample should still be 1.0
        assert np.isclose(result[15], 1.0)

    def test_very_short_chunk(self):
        chunk = np.ones(2, dtype=np.float32)
        result = apply_hann_fade_to_chunk(chunk, is_first_chunk=True, blend_samples=64)
        # fade_len = min(64, 1) = 1, just 1 sample each side
        assert result.shape == (2,)

    def test_single_sample_chunk(self):
        chunk = np.ones(1, dtype=np.float32)
        result = apply_hann_fade_to_chunk(chunk, is_first_chunk=True)
        # fade_len = min(64, 0) = 0, no fade applied
        assert np.isclose(result[0], 1.0)

    def test_empty_chunk(self):
        chunk = np.array([], dtype=np.float32)
        result = apply_hann_fade_to_chunk(chunk, is_first_chunk=True)
        assert result.size == 0

    def test_blend_samples_zero_disables_fade(self):
        chunk = np.ones(256, dtype=np.float32)
        result = apply_hann_fade_to_chunk(chunk, is_first_chunk=True, blend_samples=0)
        np.testing.assert_array_equal(result, chunk)

    def test_default_blend_samples(self):
        assert DEFAULT_BLEND_SAMPLES == 64
