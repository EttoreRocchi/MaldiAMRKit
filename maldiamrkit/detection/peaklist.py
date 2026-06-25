"""Compact peak-set containers and a raw-file peak-set loader.

:class:`PeakSet` and :class:`PeakList` are NumPy-only containers that represent
a MALDI-TOF spectrum (or a collection of spectra) as a variable-length set of
``(m/z, intensity)`` peaks.

Peak extraction is a pure per-spectrum function (no dataset-fitted state), so a
:class:`PeakList` precomputed over a whole dataset is byte-identical to one
computed inside a single cross-validation fold - caching it cannot leak. Any
fitted/aggregating step (warp reference, standardization) belongs to the
consumer's ``fit`` and is never produced or cached here; ``meta["warped"]`` is
``False`` for every list this module builds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

RANK_BY = ("intensity", "persistence", "prominence")


def _config_hash(config: dict[str, Any]) -> str:
    """Return a stable short hash of a JSON-serialisable config dict."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _array_hash(*arrays: np.ndarray) -> str:
    """Return a stable short hash of the raw bytes of one or more arrays."""
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _paths_hash(paths: Sequence[str | Path]) -> str:
    """Hash a list of files by ``(path, size, mtime)`` for cache keying."""
    h = hashlib.sha256()
    for p in paths:
        pth = Path(p)
        h.update(str(pth).encode())
        try:
            st = pth.stat()
            h.update(str(st.st_size).encode())
            h.update(str(st.st_mtime_ns).encode())
        except OSError:
            pass
    return h.hexdigest()[:16]


@dataclass(frozen=True, eq=False, repr=False)
class PeakSet:
    """One spectrum represented as a set of ``(m/z, intensity)`` peaks.

    Peaks are stored sorted by ascending ``m/z``. The container is
    permutation-agnostic: the set of peaks is what matters, not their order.

    Parameters
    ----------
    mz : array-like
        Peak m/z positions, shape ``(n_peaks,)``.
    intensity : array-like
        Peak intensities aligned with ``mz``, shape ``(n_peaks,)``.
    score : array-like, optional
        Per-peak ranking score (e.g. persistence or prominence) used by
        :meth:`top_k`. ``None`` (the default) ranks by intensity. The metric
        it represents is recorded in the owning :class:`PeakList`'s
        ``meta["rank_by"]``.
    """

    mz: np.ndarray
    intensity: np.ndarray
    score: np.ndarray | None = None

    def __post_init__(self) -> None:
        mz = np.asarray(self.mz, dtype=np.float64).ravel()
        intensity = np.asarray(self.intensity, dtype=np.float64).ravel()
        if mz.shape != intensity.shape:
            raise ValueError(
                "mz and intensity must have the same shape; "
                f"got {mz.shape} and {intensity.shape}."
            )
        score = self.score
        if score is not None:
            score = np.asarray(score, dtype=np.float64).ravel()
            if score.shape != mz.shape:
                raise ValueError(
                    "score must have the same shape as mz; "
                    f"got {score.shape} and {mz.shape}."
                )
        order = np.argsort(mz, kind="stable")
        object.__setattr__(self, "mz", mz[order])
        object.__setattr__(self, "intensity", intensity[order])
        object.__setattr__(self, "score", None if score is None else score[order])

    def __repr__(self) -> str:
        if self.n_peaks == 0:
            return "PeakSet(n_peaks=0)"
        return (
            f"PeakSet(n_peaks={self.n_peaks}, mz=[{self.mz[0]:.1f}..{self.mz[-1]:.1f}])"
        )

    @property
    def n_peaks(self) -> int:
        """Number of peaks in the set."""
        return int(self.mz.shape[0])

    def __len__(self) -> int:
        return self.n_peaks

    def top_k(self, k: int) -> "PeakSet":
        """Return the ``k`` top-ranked peaks (still m/z-sorted).

        Ranks by the per-peak :attr:`score` (e.g. persistence or prominence)
        when present, otherwise by intensity. The carried score, if any, is
        sliced alongside the kept peaks so the ranking stays consistent.
        """
        if k < 0:
            raise ValueError(f"k must be non-negative; got {k}.")
        if self.n_peaks <= k:
            return PeakSet(
                self.mz.copy(),
                self.intensity.copy(),
                None if self.score is None else self.score.copy(),
            )
        rank = self.intensity if self.score is None else self.score
        keep = np.argsort(rank, kind="stable")[::-1][:k]
        return PeakSet(
            self.mz[keep],
            self.intensity[keep],
            None if self.score is None else self.score[keep],
        )

    def as_array(self) -> np.ndarray:
        """Return an ``(n_peaks, 2)`` array with ``(m/z, intensity)`` columns."""
        return np.column_stack([self.mz, self.intensity])


class PeakList:
    """A collection of per-spectrum :class:`PeakSet` objects.

    Parameters
    ----------
    peaks : sequence of PeakSet
        One peak set per spectrum.
    index : sequence, optional
        Sample identifiers, one per peak set. Defaults to a ``RangeIndex``.
    meta : dict, optional
        Provenance metadata (e.g. ``method``, ``top_k``, ``rank_by``,
        ``mz_range``, ``warped``, ``content_hash``, ``config_hash``).
        ``warped`` defaults to ``False``.
    """

    def __init__(
        self,
        peaks: Sequence[PeakSet],
        index: Sequence[Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.peaks: list[PeakSet] = list(peaks)
        if index is None:
            self.index: pd.Index = pd.RangeIndex(len(self.peaks))
        else:
            self.index = pd.Index(index)
            if len(self.index) != len(self.peaks):
                raise ValueError(
                    f"index has {len(self.index)} entries but there are "
                    f"{len(self.peaks)} peak sets."
                )
        self.meta: dict[str, Any] = {"warped": False}
        if meta:
            self.meta.update(meta)

    def __len__(self) -> int:
        return len(self.peaks)

    def __getitem__(self, i: int | slice) -> "PeakSet | PeakList":
        if isinstance(i, slice):
            return PeakList(self.peaks[i], index=self.index[i], meta=dict(self.meta))
        return self.peaks[i]

    def __iter__(self):
        return iter(self.peaks)

    def __repr__(self) -> str:
        return (
            f"PeakList(n={len(self.peaks)}, "
            f"method={self.meta.get('method', '?')}, "
            f"top_k={self.meta.get('top_k', '?')})"
        )

    @property
    def n_peaks(self) -> np.ndarray:
        """Per-spectrum peak counts, shape ``(n_samples,)``."""
        return np.array([p.n_peaks for p in self.peaks], dtype=np.int64)

    def to_padded(self, max_peaks: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Pad to a dense ``(n, P, 2)`` array plus a boolean mask.

        Parameters
        ----------
        max_peaks : int, optional
            Padded width ``P``. Defaults to the largest per-spectrum peak
            count. Spectra with more peaks than ``P`` keep their ``P`` most
            intense peaks.

        Returns
        -------
        values : np.ndarray
            Shape ``(n_samples, P, 2)`` (``float32``); last axis is
            ``(m/z, intensity)``. Padded positions are zero.
        mask : np.ndarray
            Shape ``(n_samples, P)`` (``bool``); ``True`` marks real peaks.
        """
        counts = self.n_peaks
        cap = int(counts.max()) if counts.size and counts.max() > 0 else 0
        if max_peaks is not None:
            cap = int(max_peaks)
        n = len(self.peaks)
        values = np.zeros((n, cap, 2), dtype=np.float32)
        mask = np.zeros((n, cap), dtype=bool)
        for i, peak in enumerate(self.peaks):
            if peak.n_peaks > cap:
                peak = peak.top_k(cap)
            m = peak.n_peaks
            if m == 0:
                continue
            values[i, :m, 0] = peak.mz
            values[i, :m, 1] = peak.intensity
            mask[i, :m] = True
        return values, mask

    def save(self, path: str | Path) -> None:
        """Persist to ``<path>.npz`` (numeric arrays) + ``<path>.json`` (sidecar).

        The ``.npz`` holds the concatenated peak ``mz`` / ``intensity`` plus
        per-spectrum ``offsets``; the ``.json`` holds the sample index and the
        ``meta`` dict.
        """
        base = Path(path)
        if base.suffix in {".npz", ".json"}:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)

        counts = self.n_peaks
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        if self.peaks:
            mz_concat = np.concatenate(
                [np.asarray(p.mz, dtype=np.float64) for p in self.peaks]
            )
            int_concat = np.concatenate(
                [np.asarray(p.intensity, dtype=np.float64) for p in self.peaks]
            )
        else:
            mz_concat = np.empty(0, dtype=np.float64)
            int_concat = np.empty(0, dtype=np.float64)

        arrays = {"mz": mz_concat, "intensity": int_concat, "offsets": offsets}
        if self.peaks and all(p.score is not None for p in self.peaks):
            arrays["score"] = np.concatenate(
                [np.asarray(p.score, dtype=np.float64) for p in self.peaks]
            )
        np.savez(base.with_suffix(".npz"), **arrays)
        sidecar = {
            "index": self.index.tolist(),
            "meta": self.meta,
            "n": len(self.peaks),
        }
        base.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "PeakList":
        """Load a :class:`PeakList` from a :meth:`save`-produced file pair."""
        base = Path(path)
        if base.suffix in {".npz", ".json"}:
            base = base.with_suffix("")
        npz_path = base.with_suffix(".npz")
        json_path = base.with_suffix(".json")
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        if not json_path.exists():
            raise FileNotFoundError(json_path)

        sidecar = json.loads(json_path.read_text())
        with np.load(npz_path) as data:
            mz = data["mz"]
            intensity = data["intensity"]
            offsets = data["offsets"]
            score = data["score"] if "score" in data.files else None
        peaks = [
            PeakSet(
                mz[offsets[i] : offsets[i + 1]],
                intensity[offsets[i] : offsets[i + 1]],
                None if score is None else score[offsets[i] : offsets[i + 1]],
            )
            for i in range(int(sidecar["n"]))
        ]
        return cls(peaks, index=sidecar["index"], meta=sidecar.get("meta"))


class _PeakCache:
    """On-disk cache for stateless peak lists, keyed by content + config hash."""

    def __init__(self, cache_dir: str | Path | None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def _base(self, content_hash: str, config_hash: str) -> Path:
        return self.cache_dir / f"peaklist_{config_hash}_{content_hash}"

    def get(self, content_hash: str, config_hash: str) -> PeakList | None:
        if self.cache_dir is None:
            return None
        base = self._base(content_hash, config_hash)
        if base.with_suffix(".npz").exists() and base.with_suffix(".json").exists():
            return PeakList.load(base)
        return None

    def put(self, content_hash: str, config_hash: str, peaklist: PeakList) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        peaklist.save(self._base(content_hash, config_hash))


def _peakset_from_raw(
    path: str | Path,
    detector: Any,
    pipeline: Any,
    top_k: int,
    rank_by: str,
) -> PeakSet:
    """Load, preprocess, and detect a single raw spectrum (module-level for joblib)."""
    from ..io.readers import read_spectrum
    from ..preprocessing.pipeline import preprocess

    spectrum = preprocess(read_spectrum(path), pipeline)
    return detector.detect_peakset(
        spectrum["mass"].to_numpy(),
        spectrum["intensity"].to_numpy(),
        top_k=top_k,
        rank_by=rank_by,
    )


def create_peakset_input(
    spectra_dir: str | Path,
    *,
    sample_ids: list[str] | None = None,
    file_extension: str = ".txt",
    duplicate_strategy: str = "first",
    top_k: int = 200,
    rank_by: str = "intensity",
    detector: Any | None = None,
    pipeline: Any | None = None,
    cache_dir: str | Path | None = None,
) -> PeakList:
    """Build a :class:`PeakList` from a directory of raw spectrum files.

    Analogue of :func:`maldiamrkit.alignment.create_raw_input`: discovers raw
    spectrum files, preprocesses each one per-spectrum, detects peaks at full
    m/z resolution, and keeps the ``top_k`` peaks. This is the faithful,
    binning-free path.

    Every step is a pure per-spectrum function, so the result is **un-warped**
    (``meta["warped"]=False``) and safe to cache globally: a precompute over the
    full dataset is identical to per-fold computation and cannot leak. Apply any
    fitted alignment (see :func:`maldiamrkit.alignment.align_peaks`) downstream,
    in the consumer's fold.

    Parameters
    ----------
    spectra_dir : str or Path
        Directory containing raw spectrum files.
    sample_ids : list of str, optional
        Explicit sample IDs. If ``None``, discovered from the directory.
    file_extension : str, default=".txt"
        Spectrum file extension.
    duplicate_strategy : str, default="first"
        How to handle duplicate sample IDs; see
        :func:`maldiamrkit.alignment.create_raw_input`.
    top_k : int, default=200
        Maximum number of peaks kept per spectrum.
    rank_by : {"intensity", "persistence", "prominence"}, default="intensity"
        Ranking used to select the ``top_k`` peaks.
    detector : MaldiPeakDetector, optional
        Peak detector. Defaults to ``MaldiPeakDetector()`` (local maxima).
    pipeline : PreprocessingPipeline, optional
        Preprocessing applied to each raw spectrum. Defaults to
        ``PreprocessingPipeline.default()``.
    cache_dir : str or Path, optional
        If given, cache the resulting :class:`PeakList` keyed by a content +
        config hash and reuse it on subsequent calls.

    Returns
    -------
    PeakList
        One :class:`PeakSet` per discovered spectrum.
    """
    from ..alignment.raw_warping import create_raw_input
    from ..preprocessing.preprocessing_pipeline import PreprocessingPipeline
    from .peak_detector import MaldiPeakDetector

    if rank_by not in RANK_BY:
        raise ValueError(f"Unknown rank_by={rank_by!r}; expected one of {RANK_BY}.")
    if detector is None:
        detector = MaldiPeakDetector()
    pipe = pipeline or PreprocessingPipeline.default()

    paths_df = create_raw_input(
        spectra_dir,
        sample_ids=sample_ids,
        file_extension=file_extension,
        duplicate_strategy=duplicate_strategy,
    )
    paths = paths_df["path"].tolist()

    config = {
        "source": "create_peakset_input",
        "top_k": int(top_k),
        "rank_by": rank_by,
        "detector": detector.get_params(),
        "detector_kwargs": dict(detector.kwargs),
        "pipeline": repr(pipe),
    }
    config_hash = _config_hash(config)
    content_hash = _paths_hash(paths)

    cache = _PeakCache(cache_dir)
    cached = cache.get(content_hash, config_hash)
    if cached is not None:
        return cached

    n_jobs = int(getattr(detector, "n_jobs", 1))
    if n_jobs == 1:
        peaks = [_peakset_from_raw(p, detector, pipe, top_k, rank_by) for p in paths]
    else:
        from joblib import Parallel, delayed

        peaks = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_peakset_from_raw)(p, detector, pipe, top_k, rank_by) for p in paths
        )

    mz_min, mz_max = pipe.mz_range
    peaklist = PeakList(
        peaks,
        index=paths_df.index,
        meta={
            "method": detector.method.value,
            "top_k": int(top_k),
            "rank_by": rank_by,
            "mz_range": [float(mz_min), float(mz_max)],
            "warped": False,
            "content_hash": content_hash,
            "config_hash": config_hash,
            "source": "create_peakset_input",
        },
    )
    cache.put(content_hash, config_hash, peaklist)
    return peaklist
