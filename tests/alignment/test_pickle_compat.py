"""Pickled aligners record the maldiamrkit version and refuse pre-0.22 state."""

import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from maldiamrkit.alignment import RawWarping, Warping, create_raw_input
from tests.conftest import SPECTRA_DIR

_PARAMS_SINCE_0_22 = (
    "shift_estimator",
    "min_gain",
    "min_support",
    "n_peaks",
    "prealign",
    "renormalize",
)


@pytest.fixture(scope="module")
def binned() -> pd.DataFrame:
    x = np.arange(400)
    rows = [
        sum(np.exp(-0.5 * ((x - c - s) / 3.0) ** 2) for c in (80, 200, 320))
        for s in (0, 1, -1, 2)
    ]
    return pd.DataFrame(rows, columns=[str(2000 + 3 * i) for i in range(400)])


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    if not any(SPECTRA_DIR.glob("*.txt")):
        pytest.skip("No spectrum files found in data directory")
    return create_raw_input(SPECTRA_DIR).iloc[:3]


@pytest.fixture(params=["warping", "raw_warping"])
def fitted(request, binned, raw):
    if request.param == "warping":
        return Warping(method="shift").fit(binned), binned
    return RawWarping(method="shift").fit(raw), raw


def _dumps_with_state(monkeypatch, estimator, make_state) -> bytes:
    """Pickle ``estimator`` with the state returned by ``make_state``."""
    with monkeypatch.context() as m:
        m.setattr(type(estimator), "__getstate__", make_state)
        return pickle.dumps(estimator)


def _legacy_state(self):
    state = dict(self.__dict__)
    for param in _PARAMS_SINCE_0_22:
        state.pop(param, None)
    return state


def _unstamped_state(self):
    return dict(self.__dict__)


class TestCurrentPickles:
    def test_round_trip_without_warning(self, fitted):
        estimator, X = fitted
        assert "_maldiamrkit_version" in estimator.__getstate__()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = pickle.loads(pickle.dumps(estimator))
        assert "_maldiamrkit_version" not in loaded.__dict__
        assert "_maldiamrkit_version" not in estimator.__dict__
        pd.testing.assert_frame_equal(loaded.transform(X), estimator.transform(X))

    def test_joblib_round_trip_without_warning(self, fitted, tmp_path):
        estimator, X = fitted
        joblib.dump(estimator, tmp_path / "est.pkl")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = joblib.load(tmp_path / "est.pkl")
        pd.testing.assert_frame_equal(loaded.transform(X), estimator.transform(X))


class TestPickles021:
    def test_load_warns_and_methods_refuse(self, fitted, monkeypatch):
        estimator, X = fitted
        data = _dumps_with_state(monkeypatch, estimator, _legacy_state)
        with pytest.warns(UserWarning, match="0.21 or earlier"):
            old = pickle.loads(data)
        for call in (
            lambda: old.transform(X),
            lambda: old.fit(X),
            lambda: old.get_alignment_quality(X),
        ):
            with pytest.raises(RuntimeError, match="create and fit a new"):
                call()

    def test_repickled_old_aligner_stays_refused(self, fitted, monkeypatch):
        estimator, X = fitted
        data = _dumps_with_state(monkeypatch, estimator, _legacy_state)
        with pytest.warns(UserWarning):
            old = pickle.loads(data)
        with pytest.warns(UserWarning, match="0.21 or earlier"):
            again = pickle.loads(pickle.dumps(old))
        with pytest.raises(RuntimeError, match="create and fit a new"):
            again.transform(X)


class TestPickles022:
    def test_unstamped_pickle_warns_and_works(self, fitted, monkeypatch):
        estimator, X = fitted
        data = _dumps_with_state(monkeypatch, estimator, _unstamped_state)
        with pytest.warns(UserWarning, match="did not record"):
            loaded = pickle.loads(data)
        pd.testing.assert_frame_equal(loaded.transform(X), estimator.transform(X))

    @pytest.mark.parametrize(
        "stamp, warns", [("0.0.1", True), ("dev", True), ("9.0.0", False)]
    )
    def test_recorded_versions(self, fitted, monkeypatch, stamp, warns):
        estimator, X = fitted

        def stamped(self):
            return dict(self.__dict__, _maldiamrkit_version=stamp)

        data = _dumps_with_state(monkeypatch, estimator, stamped)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = pickle.loads(data)
        assert any(issubclass(w.category, UserWarning) for w in caught) is warns
        pd.testing.assert_frame_equal(loaded.transform(X), estimator.transform(X))


class TestAdaptivePickles022:
    """An adaptive RawWarping pickled with 0.22.0 gets its edges from its reference."""

    def test_adaptive_edges_are_rebuilt(self, raw, monkeypatch):
        estimator = RawWarping(method="shift", bin_method="adaptive").fit(raw)

        def state_022(self):
            state = dict(self.__dict__)
            state.pop("_adaptive_edges_")
            return state

        data = _dumps_with_state(monkeypatch, estimator, state_022)
        with pytest.warns(UserWarning, match="did not record"):
            loaded = pickle.loads(data)
        pd.testing.assert_frame_equal(loaded.transform(raw), estimator.transform(raw))

    def test_rebuilt_edges_that_differ_require_a_refit(self, raw, monkeypatch):
        estimator = RawWarping(method="shift", bin_method="adaptive").fit(raw)

        def state_022(self):
            state = dict(self.__dict__)
            state.pop("_adaptive_edges_")
            return state

        data = _dumps_with_state(monkeypatch, estimator, state_022)
        edges = estimator._adaptive_edges_
        monkeypatch.setattr(
            RawWarping, "_learn_adaptive_edges", lambda self: edges[::2].copy()
        )
        with pytest.warns(UserWarning, match="refit"):
            loaded = pickle.loads(data)
        with pytest.raises(RuntimeError, match="refit"):
            loaded.transform(raw)
        monkeypatch.undo()
        pd.testing.assert_frame_equal(
            loaded.fit(raw).transform(raw), estimator.transform(raw)
        )


class TestRawWarpingBinKwargs:
    def test_misspelled_adaptive_bin_kwargs_raise(self, raw):
        with pytest.raises(TypeError, match="adaptive_min_widht"):
            RawWarping(
                method="shift",
                bin_method="adaptive",
                bin_kwargs={"adaptive_min_widht": 5},
            ).fit(raw)
