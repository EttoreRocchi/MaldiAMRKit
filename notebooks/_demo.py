"""Zenodo-hosted demo dataset loader for the MaldiAMRKit notebooks.

The larger-data notebooks (``04`` evaluation, ``05`` exploration,
``06`` differential analysis, ``07`` drift monitoring) share a single
dataset: **MALDI-Kleb-AI** (Rocchi et al., 2026, Zenodo DOI
``10.5281/zenodo.17405072``), a 370 MB archive of real MALDI-TOF mass
spectra of *Klebsiella* isolates from three Italian clinical centres
(Rome, Milan, Catania) with Amikacin / Meropenem antimicrobial-
resistance annotations.

MaldiAMRKit does not need batch correction (that lives in MaldiBatchKit),
so the loader exposes the **Rome subset only** by default - a single-
site cohort of 473 spectra that keeps the notebooks self-contained and
the runtimes short. Pass ``city=None`` to get all three centres.

The helper downloads the tarball once, caches it under
``~/.cache/maldiamrkit/`` (or the directory pointed to by the
``MALDIAMRKIT_CACHE_DIR`` environment variable), extracts it, and
returns a :class:`DemoDataset` populated with:

* ``X``    -- binned feature matrix obtained via
  :class:`maldiamrkit.MaldiSet` (samples x m/z bins).
* ``meta`` -- per-sample metadata: ``City`` (acquisition centre),
  ``Species``, and the AMR labels ``Amikacin`` / ``Meropenem``
  (``R`` / ``S`` / ``I``).
* ``mz``   -- m/z axis reported by ``MaldiSet`` (bin starts in Da).
* ``maldi_set`` -- the underlying ``MaldiSet`` object.

This module lives exclusively under ``notebooks/`` and is intentionally
kept out of the installable ``maldiamrkit`` package.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    pass


__all__ = [
    "ZENODO_DOI",
    "ZENODO_RECORD_ID",
    "ZENODO_TAR_MD5",
    "ZENODO_TAR_URL",
    "DemoDataset",
    "get_cache_dir",
    "load_maldi_kleb_ai",
]


ZENODO_RECORD_ID = "17405072"
ZENODO_DOI = "10.5281/zenodo.17405072"
ZENODO_TAR_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/maldi-tof.tar?download=1"
)
ZENODO_TAR_MD5 = "c14b6c6b4210553962faa7f1dc27d275"

_DATASET_DIRNAME = "maldi-kleb-ai"
_TAR_NAME = "maldi-tof.tar"


def get_cache_dir() -> Path:
    """Resolve the root cache directory for MaldiAMRKit demo data.

    Priority:

    1. ``$MALDIAMRKIT_CACHE_DIR`` environment variable (absolute path).
    2. ``~/.cache/maldiamrkit/`` (XDG-style default, cross-platform).

    The directory is created on first access.
    """
    env = os.environ.get("MALDIAMRKIT_CACHE_DIR")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path.home() / ".cache" / "maldiamrkit"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_paths(cache_dir: Path | None = None) -> dict[str, Path]:
    root = (cache_dir or get_cache_dir()) / _DATASET_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "tar": root / _TAR_NAME,
        "extract": root / "extracted",
    }


def _md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dest: Path, *, verbose: bool = True) -> None:
    """Stream ``url`` to ``dest`` with an optional progress bar."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "maldiamrkit-demo"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        read = 0
        step = max(1 << 20, total // 50) if total else 1 << 20  # ~50 ticks
        last_mark = 0
        chunk_size = 1 << 16
        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            out.write(buf)
            read += len(buf)
            if verbose and total and read - last_mark >= step:
                pct = 100.0 * read / total
                print(f"  downloading maldi-tof.tar ... {pct:5.1f} %", end="\r")
                last_mark = read
    tmp.replace(dest)
    if verbose:
        print(" " * 60, end="\r")


def _ensure_tar(
    cache_dir: Path | None, *, force: bool = False, verbose: bool = True
) -> Path:
    paths = _dataset_paths(cache_dir)
    tar = paths["tar"]
    if force and tar.exists():
        tar.unlink()
    if tar.exists() and _md5_of(tar) == ZENODO_TAR_MD5:
        return tar
    if tar.exists():  # stale / corrupted
        tar.unlink()
    if verbose:
        print(
            f"Downloading MALDI-Kleb-AI from Zenodo (DOI {ZENODO_DOI}; "
            f"370 MB, one-shot) to {tar} ..."
        )
    _download_with_progress(ZENODO_TAR_URL, tar, verbose=verbose)
    got = _md5_of(tar)
    if got != ZENODO_TAR_MD5:
        tar.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {tar.name}: expected {ZENODO_TAR_MD5}, "
            f"got {got}. The download may be corrupted; re-run with "
            f"force_redownload=True."
        )
    return tar


def _ensure_extracted(
    cache_dir: Path | None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    paths = _dataset_paths(cache_dir)
    extract_dir = paths["extract"]
    sentinel = extract_dir / "metadata.csv"
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if sentinel.exists():
        return extract_dir
    tar = _ensure_tar(cache_dir, verbose=verbose)
    if verbose:
        print(f"Extracting {tar.name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, mode="r:*") as tf:
        tf.extractall(extract_dir, filter="data")
    if not sentinel.exists():
        raise RuntimeError(
            f"Extraction completed but {sentinel} is missing. "
            f"Is the tarball layout still 'spectra/*.txt + metadata.csv'?"
        )
    return extract_dir


@dataclass
class DemoDataset:
    """MALDI-Kleb-AI binned intensities + aligned metadata."""

    X: pd.DataFrame
    """Binned feature matrix, shape ``(n_samples, n_bins)``."""

    meta: pd.DataFrame
    """Per-sample metadata (``City``, ``Species``, ``Amikacin``,
    ``Meropenem``), indexed by ``X.index``."""

    mz: np.ndarray
    """m/z axis as reported by ``MaldiSet`` (bin-start values in Da)."""

    info: dict = field(default_factory=dict)
    """Provenance info: Zenodo DOI, MD5 checksum, loader arguments."""

    maldi_set: Any = None
    """Underlying ``maldiamrkit.MaldiSet`` instance."""

    @property
    def species(self) -> pd.Series:
        """Alias for ``meta['Species']``."""
        return self.meta["Species"]

    @property
    def city(self) -> pd.Series:
        """Alias for ``meta['City']`` (acquisition centre)."""
        return self.meta["City"]


def _filtered_metadata(
    metadata_csv: Path,
    *,
    city: str | None,
    cache_dir: Path,
) -> Path:
    """Materialise a city-filtered metadata CSV that MaldiSet can read.

    When ``city`` is ``None`` we just return the original metadata CSV.
    Otherwise we write a small filtered CSV next to the extracted
    tarball; subsequent calls reuse it.
    """
    if city is None:
        return metadata_csv
    slug = city.lower().replace(" ", "_")
    out = cache_dir / f"metadata_{slug}.csv"
    if out.exists():
        return out
    df = pd.read_csv(metadata_csv)
    if "City" not in df.columns:
        raise RuntimeError(
            f"Expected a 'City' column in {metadata_csv}; got {df.columns.tolist()}."
        )
    keep = df[df["City"].str.lower() == city.lower()].copy()
    if keep.empty:
        raise ValueError(
            f"No rows match City == {city!r}. Available cities: "
            f"{sorted(df['City'].unique().tolist())}."
        )
    keep.to_csv(out, index=False)
    return out


def load_maldi_kleb_ai(
    *,
    antibiotic: str = "Amikacin",
    city: str | None = "Rome",
    bin_width: int = 3,
    cache_dir: Path | None = None,
    force_redownload: bool = False,
    verbose: bool = False,
) -> DemoDataset:
    """Download (once) and return the MALDI-Kleb-AI demo dataset.

    Parameters
    ----------
    antibiotic : {'Amikacin', 'Meropenem'}, default='Amikacin'
        Which AMR column to expose as the primary label.
    city : {'Rome', 'Milan', 'Catania', None}, default='Rome'
        Restrict the dataset to a single acquisition centre. The
        default (``'Rome'``) is the largest sub-cohort (473 spectra)
        and is the single-site demo used throughout the notebooks
        because MaldiAMRKit on its own does not perform batch
        correction (multi-site harmonisation lives in MaldiBatchKit).
        Pass ``None`` to load all three centres.
    bin_width : int, default=3
        Bin width in Daltons forwarded to
        :meth:`maldiamrkit.MaldiSet.from_directory`.
    cache_dir : Path, optional
        Root cache directory override.
    force_redownload : bool, default=False
        Re-download the tarball and re-extract even if a valid cache
        already exists.
    verbose : bool, default=False
        Print progress for the download and extraction steps.

    Returns
    -------
    DemoDataset
        With ``X``, ``meta`` (``City`` / ``Species`` / ``Amikacin`` /
        ``Meropenem``), ``mz``, and the underlying ``maldi_set``.

    Notes
    -----
    * The dataset is real clinical data (Klebsiella isolates from
      Rome, Milan, Catania); please cite the Zenodo record if you
      reuse it. DOI: ``10.5281/zenodo.17405072``.
    * First call downloads 370 MB; subsequent calls are millisecond-
      fast (the loader re-uses the extracted spectra).
    """
    try:
        from maldiamrkit import MaldiSet
    except ImportError as exc:  # pragma: no cover - maldiamrkit is the package
        raise ImportError(
            "load_maldi_kleb_ai requires the maldiamrkit package itself."
        ) from exc

    if antibiotic not in ("Amikacin", "Meropenem"):
        raise ValueError(
            f"antibiotic must be one of 'Amikacin' / 'Meropenem', got {antibiotic!r}."
        )

    paths = _dataset_paths(cache_dir)
    extract_dir = _ensure_extracted(cache_dir, force=force_redownload, verbose=verbose)
    metadata_csv = _filtered_metadata(
        extract_dir / "metadata.csv",
        city=city,
        cache_dir=paths["root"],
    )
    spectra_dir = extract_dir / "spectra"

    ds = MaldiSet.from_directory(
        str(spectra_dir),
        str(metadata_csv),
        aggregate_by={"antibiotics": [antibiotic]},
        bin_width=bin_width,
        verbose=verbose,
    )
    X = ds.X
    meta = ds.meta.loc[X.index].copy()
    mz = np.asarray(X.columns, dtype=float)

    info = {
        "source": "Zenodo MALDI-Kleb-AI",
        "doi": ZENODO_DOI,
        "record_id": ZENODO_RECORD_ID,
        "md5_tar": ZENODO_TAR_MD5,
        "n_samples": X.shape[0],
        "n_bins": X.shape[1],
        "bin_width": bin_width,
        "antibiotic": antibiotic,
        "city": city,
        "cache_dir": str(paths["root"]),
    }
    return DemoDataset(X=X, meta=meta, mz=mz, info=info, maldi_set=ds)
