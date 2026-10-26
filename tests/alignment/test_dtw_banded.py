"""The banded DTW gives exactly the tslearn path and output."""

import contextlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maldiamrkit import MaldiSpectrum
from maldiamrkit.alignment import RawWarping, Warping, create_raw_input
from maldiamrkit.alignment import strategies as strategies_module
from maldiamrkit.alignment._dtw import _banded_path, _dtw_path
from maldiamrkit.alignment.strategies import DTWStrategy
from tests.conftest import SPECTRA_DIR


def _tslearn_path(query, reference, radius):
    from tslearn.metrics import dtw_path

    path, _ = dtw_path(
        query, reference, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius
    )
    path = np.asarray(path, dtype=np.int64)
    return path[:, 0], path[:, 1]


def _dtw_align_v0220(query, reference, radius):
    """``DTWStrategy._dtw_align`` of maldiamrkit 0.22.0 (tslearn and a loop)."""
    from tslearn.metrics import dtw_path

    path, _ = dtw_path(
        query, reference, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius
    )
    aligned_sum = np.zeros_like(reference)
    aligned_count = np.zeros_like(reference)
    for i, j in path:
        if 0 <= j < len(aligned_sum):
            aligned_sum[j] += query[i]
            aligned_count[j] += 1
    aligned = np.zeros_like(reference)
    mask = aligned_count > 0
    aligned[mask] = aligned_sum[mask] / aligned_count[mask]
    return aligned


def _assert_same_path(query, reference, radius):
    expected = _tslearn_path(query, reference, radius)
    got = _dtw_path(query, reference, radius)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


@pytest.fixture(scope="module", autouse=True)
def compiled():
    """Compile the kernel once, outside the timed (Hypothesis) examples."""
    _dtw_path(np.zeros(3), np.zeros(3), 1)


@pytest.fixture(scope="module")
def binned_rows() -> np.ndarray:
    files = sorted(SPECTRA_DIR.glob("*.txt"))
    if not files:
        pytest.skip("No spectrum files found in data directory")
    return np.vstack(
        [MaldiSpectrum(f).bin(3).binned["intensity"].to_numpy() for f in files]
    )


class TestPathIdentity:
    @pytest.mark.parametrize(
        "radius",
        [
            pytest.param(0, marks=pytest.mark.slow),
            pytest.param(1, marks=pytest.mark.slow),
            10,
            pytest.param(100, marks=pytest.mark.slow),
        ],
    )
    def test_real_rows(self, binned_rows, radius):
        reference = np.median(binned_rows, axis=0)
        for row in binned_rows:
            _assert_same_path(row, reference, radius)

    @pytest.mark.slow
    def test_real_rows_wide_band(self, binned_rows):
        reference = np.median(binned_rows, axis=0)
        n = len(reference)
        for row in binned_rows[:3]:
            for radius in (n - 1, n + 5):
                _assert_same_path(row, reference, radius)

    @pytest.mark.parametrize("n", [1, 2, 3, 10])
    @pytest.mark.parametrize("radius", [0, 1, 5, 50])
    def test_engineered_ties(self, n, radius):
        rng = np.random.default_rng(n * 100 + radius)
        cases = [
            (np.zeros(n), np.zeros(n)),
            (np.ones(n), np.zeros(n)),
            (np.arange(n, dtype=float), np.arange(n, dtype=float)[::-1].copy()),
            (np.round(rng.random(n), 2), np.round(rng.random(n), 2)),
            (rng.integers(0, 3, n).astype(float), rng.integers(0, 3, n).astype(float)),
        ]
        same = rng.random(n)
        cases.append((same, same.copy()))
        for query, reference in cases:
            _assert_same_path(query, reference, radius)

    @settings(max_examples=300, deadline=None)
    @given(
        st.integers(1, 40).flatmap(
            lambda n: st.tuples(
                st.lists(st.integers(0, 3), min_size=n, max_size=n),
                st.lists(st.integers(0, 3), min_size=n, max_size=n),
                st.integers(0, 45),
            )
        )
    )
    def test_property_heavy_ties(self, case):
        query, reference, radius = case
        _assert_same_path(np.array(query), np.array(reference), radius)

    def test_python_kernel_matches_compiled(self):
        rng = np.random.default_rng(3)
        for n, radius in [(1, 0), (5, 1), (12, 3), (30, 40)]:
            query = rng.integers(0, 3, n).astype(float)
            reference = rng.integers(0, 3, n).astype(float)
            compiled = _banded_path(query, reference, radius)
            python = _banded_path.py_func(query, reference, radius)
            np.testing.assert_array_equal(compiled[0], python[0])
            np.testing.assert_array_equal(compiled[1], python[1])


class TestAlignIdentity:
    @pytest.mark.parametrize(
        "query_dtype, reference_dtype",
        [
            (np.float64, np.float64),
            (np.float32, np.float32),
            (np.float32, np.float64),
            (np.float64, np.float32),
        ],
    )
    def test_dtw_align_matches_v0220(self, binned_rows, query_dtype, reference_dtype):
        reference = np.median(binned_rows, axis=0).astype(reference_dtype)
        strategy = DTWStrategy(10)
        for row in binned_rows[:10]:
            query = row.astype(query_dtype)
            np.testing.assert_array_equal(
                strategy._dtw_align(query, reference),
                _dtw_align_v0220(query, reference, 10),
            )


class TestDispatch:
    @pytest.fixture()
    def tslearn_calls(self, monkeypatch):
        import tslearn.metrics

        calls = []
        original = tslearn.metrics.dtw_path

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(tslearn.metrics, "dtw_path", spy)
        return calls

    def test_finite_equal_length_uses_the_band(self, tslearn_calls):
        DTWStrategy(2)._dtw_align(np.arange(10.0), np.arange(10.0))
        assert not tslearn_calls

    @pytest.mark.parametrize(
        "query, reference, radius",
        [
            (np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0, 3.0]), 1),
            (np.arange(5.0), np.arange(6.0), 1),
            (np.arange(10.0), np.arange(10.0), -1),
            (np.arange(10.0), np.arange(10.0), 1.5),
            (np.arange(10.0), np.arange(10.0), True),
            (np.arange(10.0), np.arange(10.0), 5),
        ],
    )
    def test_other_inputs_use_tslearn(self, tslearn_calls, query, reference, radius):
        with contextlib.suppress(Exception):
            DTWStrategy(radius)._dtw_align(query, reference)
        assert tslearn_calls


class TestEstimatorsMatchTslearn:
    def test_warping(self, binned_rows, monkeypatch):
        X = pd.DataFrame(binned_rows[:8])
        banded = Warping(method="dtw").fit(X).transform(X)
        monkeypatch.setattr(strategies_module, "_use_banded_dtw", lambda *a: False)
        expected = Warping(method="dtw").fit(X).transform(X)
        pd.testing.assert_frame_equal(banded, expected, check_exact=True)

    def test_raw_warping(self, monkeypatch):
        X = create_raw_input(SPECTRA_DIR).iloc[:3]
        banded = RawWarping(method="dtw", interp_step=3.0).fit(X).transform(X)
        monkeypatch.setattr(strategies_module, "_use_banded_dtw", lambda *a: False)
        expected = RawWarping(method="dtw", interp_step=3.0).fit(X).transform(X)
        pd.testing.assert_frame_equal(banded, expected, check_exact=True)
