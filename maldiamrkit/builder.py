"""Build DRIAMS-like dataset directories from raw spectra and metadata.

Given a directory of raw MALDI-TOF spectrum files and a metadata CSV, this
module produces a structured dataset directory following the DRIAMS
convention:

.. code-block:: text

    output/
    ├── raw/
    │   ├── {year}/          # optional year subfolders
    │   │   ├── {id}.txt
    ├── preprocessed/
    │   ├── {year}/
    │   │   ├── {id}.txt
    ├── binned_6000/
    │   ├── {year}/
    │   │   ├── {id}.txt
    └── id/
        ├── {year}/
        │   └── {year}_clean.csv

Additional processing outputs (different pipelines or bin widths) can be
produced alongside the defaults via :class:`ProcessingHandler`.

Examples
--------
>>> from maldiamrkit.builder import build_driams_dataset, ProcessingHandler
>>> report = build_driams_dataset("spectra/", "meta.csv", "output/")
>>> report = build_driams_dataset(
...     "spectra/", "meta.csv", "output/",
...     year_column="acquisition_date",
...     extra_handlers=[
...         ProcessingHandler("preprocessed_sqrt", "preprocessed",
...                           pipeline=sqrt_pipeline),
...         ProcessingHandler("binned_3000", "binned", bin_width=6),
...     ],
... )
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .io.readers import read_spectrum
from .preprocessing.binning import _uniform_edges, bin_spectrum
from .preprocessing.pipeline import preprocess
from .preprocessing.preprocessing_pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProcessingHandler:
    """Define an additional processing output folder.

    Each handler produces one output folder containing spectra processed
    with the specified pipeline and (optionally) binned with the given
    bin width.

    Parameters
    ----------
    folder_name : str
        Name of the output folder (e.g. ``"preprocessed_sqrt"``).
    kind : str
        Either ``"preprocessed"`` (apply pipeline only) or ``"binned"``
        (apply pipeline then bin).
    pipeline : PreprocessingPipeline or None
        Pipeline to apply.  ``None`` uses the default pipeline.
    bin_width : int or float
        Bin width in Daltons.  Only used when ``kind="binned"``.

    Examples
    --------
    >>> handler = ProcessingHandler("preprocessed_sqrt", "preprocessed",
    ...                             pipeline=sqrt_pipeline)
    >>> handler = ProcessingHandler("binned_3000", "binned", bin_width=6)
    """

    folder_name: str
    kind: str
    pipeline: PreprocessingPipeline | None = None
    bin_width: int | float = 3

    def __post_init__(self) -> None:
        if self.kind not in ("preprocessed", "binned"):
            raise ValueError(
                f"Invalid kind '{self.kind}'. Must be 'preprocessed' or 'binned'."
            )

    def to_dict(self) -> dict:
        """Serialize to a dictionary suitable for JSON/YAML.

        Returns
        -------
        dict
            Dictionary representation.  The ``pipeline`` field is stored
            as its own dict (see
            :meth:`PreprocessingPipeline.to_dict`) or ``None``.
        """
        return {
            "folder_name": self.folder_name,
            "kind": self.kind,
            "pipeline": self.pipeline.to_dict() if self.pipeline is not None else None,
            "bin_width": self.bin_width,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProcessingHandler:
        """Reconstruct from a dictionary.

        The ``pipeline`` value can be:

        * ``None`` - uses the default pipeline.
        * A ``dict`` - parsed via :meth:`PreprocessingPipeline.from_dict`.
        * A ``str`` - treated as a file path (JSON or YAML).

        Parameters
        ----------
        d : dict
            Dictionary as produced by :meth:`to_dict`, or a simplified
            form from a YAML/JSON config file.

        Returns
        -------
        ProcessingHandler
            Reconstructed handler.
        """
        pipeline_val = d.get("pipeline")
        if pipeline_val is None:
            pipeline = None
        elif isinstance(pipeline_val, dict):
            pipeline = PreprocessingPipeline.from_dict(pipeline_val)
        elif isinstance(pipeline_val, str):
            path = Path(pipeline_val)
            if path.suffix in (".yaml", ".yml"):
                pipeline = PreprocessingPipeline.from_yaml(path)
            else:
                pipeline = PreprocessingPipeline.from_json(path)
        else:
            raise TypeError(f"Unsupported pipeline value type: {type(pipeline_val)}")

        return cls(
            folder_name=d["folder_name"],
            kind=d["kind"],
            pipeline=pipeline,
            bin_width=d.get("bin_width", 3),
        )


@dataclasses.dataclass
class BuildReport:
    """Summary of a DRIAMS dataset build.

    Attributes
    ----------
    total : int
        Number of spectra attempted.
    succeeded : int
        Number successfully processed.
    failed : int
        Number that failed.
    failed_ids : list of str
        IDs of spectra that failed.
    output_dir : Path
        Root of the output dataset.
    folders_created : list of str
        Names of all processing folders created.
    """

    total: int
    succeeded: int
    failed: int
    failed_ids: list[str]
    output_dir: Path
    folders_created: list[str]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_driams_spectrum(df: pd.DataFrame, path: Path) -> None:
    """Save mass/intensity DataFrame in DRIAMS space-separated format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        df[["mass", "intensity"]].values.astype(float),
        header="mass intensity",
        comments="# ",
        fmt="%.6f",
    )


def _save_driams_binned(binned_df: pd.DataFrame, path: Path) -> None:
    """Save binned spectrum as ``bin_index binned_intensity``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.arange(len(binned_df))
    intensities = binned_df["intensity"].values.astype(float)
    data = np.column_stack([indices, intensities])
    np.savetxt(
        path,
        data,
        header="bin_index binned_intensity",
        comments="",
        fmt=["%d", "%.17g"],
    )


def _extract_year(value: object) -> str:
    """Extract a four-digit year string from various input types.

    Supports:
    - Strings like ``"2015-01-12"`` or ``"2015"``
    - ``datetime`` / ``Timestamp`` objects
    - Integers / floats (e.g. ``2015`` or ``2015.0``)

    Parameters
    ----------
    value : object
        Value to extract year from.

    Returns
    -------
    str
        Four-digit year string.

    Raises
    ------
    ValueError
        If the year cannot be extracted.
    """
    if hasattr(value, "year"):
        # datetime / Timestamp
        return str(value.year)
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, str):
        # Try "YYYY-..." or plain "YYYY"
        part = value.strip().split("-")[0].split("/")[0]
        if part.isdigit() and len(part) == 4:
            return part
    raise ValueError(f"Cannot extract year from {value!r}")


def _process_single_spectrum(
    src_path: Path,
    output_dir: Path,
    default_pipeline: PreprocessingPipeline,
    default_bin_width: int | float,
    default_binned_folder: str,
    extra_handlers: list[ProcessingHandler],
    year: str | None,
    mz_min: int,
    mz_max: int,
) -> str | None:
    """Process one spectrum through all outputs.  Returns ID on success."""
    spec_id = src_path.stem
    year_sub = f"{year}/" if year else ""

    try:
        raw_df = read_spectrum(src_path)

        # 1. Save raw
        _save_driams_spectrum(raw_df, output_dir / "raw" / f"{year_sub}{spec_id}.txt")

        # 2. Default preprocessing + save
        preprocessed = preprocess(raw_df, default_pipeline)
        _save_driams_spectrum(
            preprocessed,
            output_dir / "preprocessed" / f"{year_sub}{spec_id}.txt",
        )

        # 3. Default binning + save
        binned, _ = bin_spectrum(
            preprocessed,
            mz_min=mz_min,
            mz_max=mz_max,
            bin_width=default_bin_width,
        )
        _save_driams_binned(
            binned,
            output_dir / default_binned_folder / f"{year_sub}{spec_id}.txt",
        )

        # 4. Extra handlers
        for handler in extra_handlers:
            h_pipeline = handler.pipeline or default_pipeline
            h_preprocessed = preprocess(raw_df, h_pipeline)

            if handler.kind == "preprocessed":
                _save_driams_spectrum(
                    h_preprocessed,
                    output_dir / handler.folder_name / f"{year_sub}{spec_id}.txt",
                )
            elif handler.kind == "binned":
                h_binned, _ = bin_spectrum(
                    h_preprocessed,
                    mz_min=mz_min,
                    mz_max=mz_max,
                    bin_width=handler.bin_width,
                )
                _save_driams_binned(
                    h_binned,
                    output_dir / handler.folder_name / f"{year_sub}{spec_id}.txt",
                )

        return spec_id
    except Exception as exc:
        logger.warning("Failed to process %s: %s", spec_id, exc)
        return None


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    spectra_dir: Path,
    metadata_csv: Path,
    year_column: str | None,
) -> tuple[list[Path], pd.DataFrame]:
    """Validate inputs and return spectrum files and metadata."""
    spectrum_files = sorted(spectra_dir.glob("*.txt"))
    if not spectrum_files:
        raise ValueError(f"No .txt spectrum files found in {spectra_dir}")

    meta = pd.read_csv(metadata_csv)
    if "ID" not in meta.columns:
        raise ValueError(
            f"Metadata CSV must have an 'ID' column. "
            f"Found columns: {list(meta.columns)}"
        )

    if year_column is not None and year_column not in meta.columns:
        raise ValueError(
            f"year_column '{year_column}' not found in metadata. "
            f"Available: {list(meta.columns)}"
        )

    return spectrum_files, meta


def _compute_folder_layout(
    mz_min: float,
    mz_max: float,
    bin_width: int | float,
    extra_handlers: list[ProcessingHandler],
) -> tuple[str, list[str]]:
    """Compute folder names and check for duplicates."""
    n_bins = len(_uniform_edges(mz_min, mz_max, bin_width)) - 1
    default_binned_folder = f"binned_{n_bins}"

    all_folders = ["raw", "preprocessed", default_binned_folder] + [
        h.folder_name for h in extra_handlers
    ]
    seen: set[str] = set()
    for folder in all_folders:
        if folder in seen:
            raise ValueError(
                f"Duplicate folder name '{folder}'. "
                f"Ensure default and extra handler folder names are unique."
            )
        seen.add(folder)

    return default_binned_folder, all_folders


def _match_ids(
    meta: pd.DataFrame,
    spectrum_files: list[Path],
) -> tuple[set[str], dict[str, Path]]:
    """Intersect metadata IDs with spectrum files."""
    meta_ids = set(meta["ID"].astype(str))
    file_map = {f.stem: f for f in spectrum_files}
    file_ids = set(file_map.keys())

    matched_ids = meta_ids & file_ids
    meta_only = meta_ids - file_ids
    files_only = file_ids - meta_ids

    if meta_only:
        logger.warning(
            "%d IDs in metadata but not in spectra dir: %s",
            len(meta_only),
            sorted(meta_only)[:5],
        )
    if files_only:
        logger.warning(
            "%d spectrum files not in metadata: %s",
            len(files_only),
            sorted(files_only)[:5],
        )

    if not matched_ids:
        raise ValueError("No matching IDs between metadata and spectrum files.")

    return matched_ids, file_map


def _build_year_map(
    meta: pd.DataFrame,
    matched_ids: set[str],
    year_column: str | None,
) -> tuple[dict[str, str] | None, set[str]]:
    """Build mapping from spectrum ID to year string."""
    if year_column is None:
        return None, set()

    matched_meta = meta[meta["ID"].astype(str).isin(matched_ids)]
    year_map = dict(
        zip(
            matched_meta["ID"].astype(str),
            matched_meta[year_column].apply(_extract_year),
            strict=True,
        )
    )
    years = set(year_map.values())

    return year_map, years


def _create_directory_skeleton(
    output_dir: Path,
    all_folders: list[str],
    years: set[str],
) -> None:
    """Create the output directory tree."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder in all_folders:
        if years:
            for year in years:
                (output_dir / folder / year).mkdir(parents=True, exist_ok=True)
        else:
            (output_dir / folder).mkdir(parents=True, exist_ok=True)

    if years:
        for year in years:
            (output_dir / "id" / year).mkdir(parents=True, exist_ok=True)
    else:
        (output_dir / "id").mkdir(parents=True, exist_ok=True)


def _write_metadata(
    meta: pd.DataFrame,
    matched_ids: set[str],
    output_dir: Path,
    year_map: dict[str, str] | None,
    years: set[str],
    year_column: str | None,
    id_column: str,
    name: str,
) -> None:
    """Write metadata CSV(s) to the output directory."""
    meta_matched = meta[meta["ID"].astype(str).isin(matched_ids)].copy()
    meta_matched = meta_matched.rename(columns={"ID": id_column})

    if year_column is not None and years:
        meta_matched["_year"] = meta_matched[id_column].astype(str).map(year_map)
        for year in sorted(years):
            year_meta = meta_matched[meta_matched["_year"] == year].drop(
                columns=["_year"]
            )
            year_meta.to_csv(
                output_dir / "id" / year / f"{year}_clean.csv", index=False
            )
        meta_matched = meta_matched.drop(columns=["_year"])
    else:
        meta_matched.to_csv(output_dir / "id" / f"{name}_clean.csv", index=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_driams_dataset(
    spectra_dir: str | Path,
    metadata_csv: str | Path,
    output_dir: str | Path,
    *,
    name: str | None = None,
    id_column: str = "code",
    year_column: str | None = None,
    pipeline: PreprocessingPipeline | None = None,
    bin_width: int | float = 3,
    extra_handlers: list[ProcessingHandler] | None = None,
    n_jobs: int = -1,
    on_error: str = "warn",
) -> BuildReport:
    """Build a DRIAMS-like dataset directory from raw spectra and metadata.

    Parameters
    ----------
    spectra_dir : str or Path
        Directory containing raw ``.txt`` spectrum files.
    metadata_csv : str or Path
        CSV file with at least an ``ID`` column matching spectrum
        filenames (without extension).
    output_dir : str or Path
        Root directory for the output dataset.
    name : str or None
        Dataset name used for metadata filenames.  Defaults to the
        ``output_dir`` directory name.
    id_column : str, default="code"
        Column name for the spectrum identifier in the output metadata
        CSV.  The input CSV must use ``ID``; this controls how it is
        renamed in the output.
    year_column : str or None
        Metadata column to extract the year from.  When set, spectra
        are organised into year-based subfolders and metadata is split
        per year.  Supports date strings, datetime objects, and
        integers.
    pipeline : PreprocessingPipeline or None
        Preprocessing pipeline for the default output.  ``None`` uses
        :meth:`PreprocessingPipeline.default`.
    bin_width : int or float, default=3
        Bin width in Daltons for the default binned output.
    extra_handlers : list of ProcessingHandler or None
        Additional processing outputs.  Each handler produces its own
        named folder.
    n_jobs : int, default=-1
        Number of parallel jobs (``-1`` for all cores).
    on_error : str, default="warn"
        How to handle per-spectrum failures: ``"warn"`` logs and
        continues, ``"raise"`` re-raises, ``"skip"`` silently skips.

    Returns
    -------
    BuildReport
        Summary of the build.

    Raises
    ------
    ValueError
        If inputs are invalid (empty directory, missing ``ID`` column,
        duplicate folder names, etc.).

    Examples
    --------
    >>> from maldiamrkit.builder import build_driams_dataset
    >>> report = build_driams_dataset("spectra/", "meta.csv", "output/")
    >>> report.succeeded
    100
    """
    spectra_dir = Path(spectra_dir)
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)
    extra_handlers = list(extra_handlers) if extra_handlers else []
    pipeline = pipeline or PreprocessingPipeline.default()
    name = name or output_dir.name

    spectrum_files, meta = _validate_inputs(spectra_dir, metadata_csv, year_column)
    mz_min, mz_max = pipeline.mz_range
    default_binned_folder, all_folders = _compute_folder_layout(
        mz_min,
        mz_max,
        bin_width,
        extra_handlers,
    )
    matched_ids, file_map = _match_ids(meta, spectrum_files)
    year_map, years = _build_year_map(meta, matched_ids, year_column)
    _create_directory_skeleton(output_dir, all_folders, years)
    _write_metadata(
        meta,
        matched_ids,
        output_dir,
        year_map,
        years,
        year_column,
        id_column,
        name,
    )

    sorted_ids = sorted(matched_ids)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_single_spectrum)(
            file_map[sid],
            output_dir,
            pipeline,
            bin_width,
            default_binned_folder,
            extra_handlers,
            year_map[sid] if year_map else None,
            mz_min,
            mz_max,
        )
        for sid in sorted_ids
    )

    succeeded_ids = [r for r in results if r is not None]
    failed_ids = [sid for sid, r in zip(sorted_ids, results, strict=True) if r is None]

    if on_error == "raise" and failed_ids:
        raise RuntimeError(f"{len(failed_ids)} spectra failed to process: {failed_ids}")

    return BuildReport(
        total=len(sorted_ids),
        succeeded=len(succeeded_ids),
        failed=len(failed_ids),
        failed_ids=failed_ids,
        output_dir=output_dir,
        folders_created=all_folders,
    )
