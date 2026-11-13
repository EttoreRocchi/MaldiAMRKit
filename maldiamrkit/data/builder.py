"""Build standardised datasets from any supported input layout.

Given an :class:`InputLayout` and an output directory, the
:class:`DatasetBuilder` reads raw spectra, applies preprocessing and
binning, and writes the results to a standardised folder structure
that can be loaded by :class:`DatasetLoader`.

Output structure::

    output_dir/
    ├── raw/{year}/{id}.txt
    ├── preprocessed/{year}/{id}.txt
    ├── binned_{N}/{year}/{id}.txt
    └── id/{year}/{year}_clean.csv
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..io.readers import read_spectrum
from ..preprocessing.binning import _uniform_edges, bin_spectrum
from ..preprocessing.pipeline import preprocess
from ..preprocessing.preprocessing_pipeline import PreprocessingPipeline
from .input_layouts import InputLayout
from .site_info import (
    BuildInfo,
    SiteInfo,
    _current_iso_utc,
    write_site_info,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProcessingHandler:
    """Define an additional processing output folder.

    Parameters
    ----------
    folder_name : str
        Name of the output folder (e.g. ``"preprocessed_sqrt"``).
    kind : str
        Either ``"preprocessed"`` or ``"binned"``.
    pipeline : PreprocessingPipeline or None
        Pipeline to apply.  ``None`` uses the default.
    bin_width : int or float
        Bin width in Daltons (only used when ``kind="binned"``).
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
        """Serialize to a dictionary."""
        return {
            "folder_name": self.folder_name,
            "kind": self.kind,
            "pipeline": self.pipeline.to_dict() if self.pipeline is not None else None,
            "bin_width": self.bin_width,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProcessingHandler:
        """Reconstruct from a dictionary."""
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
    """Summary of a dataset build.

    Attributes
    ----------
    total : int
        Number of spectra in both the inputs and the metadata, including
        those not built because their ID or input is ambiguous.
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


def _save_spectrum(df: pd.DataFrame, path: Path) -> None:
    """Save mass/intensity DataFrame in space-separated format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        df[["mass", "intensity"]].values.astype(float),
        header="mass intensity",
        comments="# ",
        fmt="%.17g",
    )


def _save_binned(binned_df: pd.DataFrame, path: Path) -> None:
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


def _process_single_spectrum(
    src_path: Path,
    spec_id: str,
    output_dir: Path,
    default_pipeline: PreprocessingPipeline,
    default_bin_width: int | float,
    default_binned_folder: str,
    extra_handlers: list[ProcessingHandler],
    year: str | None,
    mz_min: int,
    mz_max: int,
    bruker_source: str,
) -> str | None:
    """Process one spectrum through all outputs.  Returns ID on success."""
    year_sub = f"{year}/" if year else ""

    try:
        raw_df = read_spectrum(src_path, bruker_source=bruker_source)

        _save_spectrum(raw_df, output_dir / "raw" / f"{year_sub}{spec_id}.txt")

        preprocessed = preprocess(raw_df, default_pipeline)
        _save_spectrum(
            preprocessed,
            output_dir / "preprocessed" / f"{year_sub}{spec_id}.txt",
        )

        binned, _ = bin_spectrum(
            preprocessed,
            mz_min=mz_min,
            mz_max=mz_max,
            bin_width=default_bin_width,
        )
        _save_binned(
            binned,
            output_dir / default_binned_folder / f"{year_sub}{spec_id}.txt",
        )

        for handler in extra_handlers:
            h_pipeline = handler.pipeline or default_pipeline
            h_preprocessed = preprocess(raw_df, h_pipeline)

            if handler.kind == "preprocessed":
                _save_spectrum(
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
                _save_binned(
                    h_binned,
                    output_dir / handler.folder_name / f"{year_sub}{spec_id}.txt",
                )

        return spec_id
    except Exception as exc:
        logger.warning("Failed to process %s: %s", spec_id, exc)
        return None


class DatasetBuilder:
    """Build a standardised dataset from any supported input layout.

    Parameters
    ----------
    layout : InputLayout
        Input data layout adapter.
    output_dir : str or Path
        Root directory for the standardised output.
    name : str or None
        Dataset name (defaults to ``output_dir`` name).
    id_column : str, default="code"
        Column name for the spectrum identifier in the output metadata.
    pipeline : PreprocessingPipeline or None
        Preprocessing pipeline.  ``None`` uses the default.
    bin_width : int or float, default=3
        Bin width in Daltons for the default binned output.
    extra_handlers : list of ProcessingHandler or None
        Additional processing outputs.
    metadata_dir : str, default="id"
        Subdirectory name for metadata CSV output.
    metadata_suffix : str, default="_clean.csv"
        Filename suffix for metadata CSV output.
    n_jobs : int, default=-1
        Number of parallel workers.
    on_error : str, default="warn"
        Error handling: ``"warn"``, ``"raise"``, or ``"skip"``.
        With ``"raise"``, an ambiguous ID or input (see Notes) raises
        before anything is written.
    bruker_source : {"1r", "fid"}, default="1r"
        For Bruker directories, which binary is read into ``raw/``:
        ``"1r"`` (``pdata/1/1r``) or ``"fid"`` (the raw free induction
        decay). A spectrum whose binary is missing fails. Recorded in
        ``site_info.json``.

    Notes
    -----
    A spectrum ID found in several inputs (e.g. ``2018/x.txt`` and
    ``2019/x.txt``), and an input named by several IDs (one Bruker
    directory named by two identifiers), cannot be built unambiguously:
    none of them is built, and each of their IDs is reported once in
    ``BuildReport.failed_ids``.

    Examples
    --------
    >>> from maldiamrkit.data import DatasetBuilder, FlatLayout
    >>> layout = FlatLayout("spectra/", "meta.csv")
    >>> builder = DatasetBuilder(layout, "output/")
    >>> report = builder.build()
    """

    def __init__(
        self,
        layout: InputLayout,
        output_dir: str | Path,
        *,
        name: str | None = None,
        id_column: str = "code",
        pipeline: PreprocessingPipeline | None = None,
        bin_width: int | float = 3,
        extra_handlers: list[ProcessingHandler] | None = None,
        metadata_dir: str = "id",
        metadata_suffix: str = "_clean.csv",
        n_jobs: int = -1,
        on_error: str = "warn",
        bruker_source: str = "1r",
    ) -> None:
        if bruker_source not in ("1r", "fid"):
            raise ValueError(
                f"Invalid bruker_source {bruker_source!r}; expected '1r' or 'fid'."
            )
        self.layout = layout
        self.output_dir = Path(output_dir)
        self.name = name or self.output_dir.name
        self.id_column = id_column
        self.pipeline = pipeline or PreprocessingPipeline.default()
        self.bin_width = bin_width
        self.extra_handlers = list(extra_handlers) if extra_handlers else []
        self.metadata_dir = metadata_dir
        self.metadata_suffix = metadata_suffix
        self.n_jobs = n_jobs
        self.on_error = on_error
        self.bruker_source = bruker_source

    def build(self) -> BuildReport:
        """Execute the build pipeline.

        Returns
        -------
        BuildReport
            Summary of the build.
        """
        spectrum_paths = self.layout.discover_spectra()
        meta = self.layout.discover_metadata()

        matched_ids, id_to_path, collided = self._match_ids(spectrum_paths, meta)
        if collided and self.on_error == "raise":
            raise RuntimeError(
                f"{len(collided)} spectrum IDs are found in several inputs or share "
                f"an input with another ID: {collided[:5]}"
            )

        mz_min, mz_max = self.pipeline.mz_range
        n_bins = len(_uniform_edges(mz_min, mz_max, self.bin_width)) - 1
        default_binned_folder = f"binned_{n_bins}"

        all_folders = ["raw", "preprocessed", default_binned_folder] + [
            h.folder_name for h in self.extra_handlers
        ]
        seen: set[str] = set()
        for folder in all_folders:
            if folder in seen:
                raise ValueError(f"Duplicate folder name '{folder}'.")
            seen.add(folder)

        year_map, years = self._build_year_map(matched_ids)
        self._create_skeleton(years, all_folders)
        self._write_metadata(meta, matched_ids, year_map, years)

        # Process spectra in parallel
        sorted_ids = sorted(matched_ids)
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_process_single_spectrum)(
                id_to_path[sid],
                sid,
                self.output_dir,
                self.pipeline,
                self.bin_width,
                default_binned_folder,
                self.extra_handlers,
                year_map.get(sid),
                mz_min,
                mz_max,
                bruker_source=self.bruker_source,
            )
            for sid in sorted_ids
        )

        succeeded_ids = [r for r in results if r is not None]
        failed_ids = sorted(
            [sid for sid, r in zip(sorted_ids, results, strict=True) if r is None]
            + collided
        )
        n_total = len(sorted_ids) + len(collided)

        if self.on_error == "raise" and failed_ids:
            raise RuntimeError(f"{len(failed_ids)} spectra failed: {failed_ids}")

        self._write_manifest(
            all_folders=all_folders,
            mz_min=mz_min,
            mz_max=mz_max,
            n_total=n_total,
            n_succeeded=len(succeeded_ids),
            n_failed=len(failed_ids),
            bruker_inputs=Path(spectrum_paths[0]).is_dir(),
        )

        return BuildReport(
            total=n_total,
            succeeded=len(succeeded_ids),
            failed=len(failed_ids),
            failed_ids=failed_ids,
            output_dir=self.output_dir,
            folders_created=all_folders,
        )

    def _write_manifest(
        self,
        *,
        all_folders: list[str],
        mz_min: float,
        mz_max: float,
        n_total: int,
        n_succeeded: int,
        n_failed: int,
        bruker_inputs: bool,
    ) -> None:
        """Write ``site_info.json`` describing this build.

        Failures are logged but never raise: a missing manifest is
        tolerated by readers, so a write error should not invalidate
        an otherwise-successful build.
        """
        from .. import __version__ as _maldiamrkit_version

        dup_attr = getattr(self.layout, "duplicate_strategy", None)
        dup_str: str | None
        if dup_attr is None:
            dup_str = None
        else:
            dup_str = getattr(dup_attr, "value", None) or str(dup_attr)

        manifest = SiteInfo(
            id_column=self.id_column,
            metadata_dir=self.metadata_dir,
            metadata_suffix=self.metadata_suffix,
            spectrum_ext=".txt",
            spectra_folders=list(all_folders),
            mz_range=(float(mz_min), float(mz_max)),
            bin_width=float(self.bin_width),
            build_info=BuildInfo(
                maldiamrkit_version=str(_maldiamrkit_version),
                created_at=_current_iso_utc(),
                source_layout=type(self.layout).__name__,
                duplicate_strategy=dup_str,
                n_total_spectra=int(n_total),
                n_succeeded=int(n_succeeded),
                n_failed=int(n_failed),
                bruker_source=self.bruker_source if bruker_inputs else None,
                metadata_rows=self.layout._metadata_rows,
            ),
        )

        try:
            write_site_info(self.output_dir, manifest)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write site_info.json: %s", exc)

    def _match_ids(
        self, spectrum_paths: list[Path], meta: pd.DataFrame
    ) -> tuple[set[str], dict[str, Path], list[str]]:
        """Match spectrum IDs with metadata and log mismatches.

        Returns the IDs to build, the input of each, and the sorted
        metadata IDs left out because they name several inputs or share
        their input with another ID.
        """
        paths_by_id: dict[str, list[Path]] = defaultdict(list)
        ids_by_path: dict[Path, list[str]] = defaultdict(list)
        for sid, path in self.layout._spectrum_ids(spectrum_paths):
            if path not in paths_by_id[sid]:
                paths_by_id[sid].append(path)
            if sid not in ids_by_path[path]:
                ids_by_path[path].append(sid)

        meta_ids = set(meta["ID"].astype(str))
        spec_ids = set(paths_by_id)
        colliding: set[str] = set()
        for sid, paths in paths_by_id.items():
            if len(paths) > 1 and sid in meta_ids:
                logger.warning(
                    "Spectrum ID '%s' names %d input files (%s): none is built.",
                    sid,
                    len(paths),
                    ", ".join(self._display_path(p) for p in paths),
                )
                colliding.add(sid)
        for path, ids in ids_by_path.items():
            named = [sid for sid in ids if sid in meta_ids]
            if len(named) > 1:
                logger.warning(
                    "Input %s is named by %d spectrum IDs (%s): none is built.",
                    self._display_path(path),
                    len(named),
                    ", ".join(named),
                )
                colliding.update(named)

        if not meta_ids & spec_ids:
            raise ValueError("No matching IDs between metadata and spectra.")
        matched_ids = (meta_ids & spec_ids) - colliding
        id_to_path = {sid: paths_by_id[sid][0] for sid in matched_ids}

        meta_only = meta_ids - spec_ids
        spec_only = spec_ids - meta_ids
        if meta_only:
            logger.warning(
                "%d IDs in metadata but not in spectra: %s",
                len(meta_only),
                sorted(meta_only)[:5],
            )
        if spec_only:
            logger.warning(
                "%d spectra not in metadata: %s",
                len(spec_only),
                sorted(spec_only)[:5],
            )
        return matched_ids, id_to_path, sorted(colliding)

    def _display_path(self, path: Path) -> str:
        """Show *path* relative to the layout's input folder when possible."""
        base = getattr(self.layout, "spectra_dir", None) or getattr(
            self.layout, "root_dir", None
        )
        if base is not None:
            try:
                return Path(path).relative_to(base).as_posix()
            except ValueError:
                pass
        return str(path)

    def _build_year_map(
        self, matched_ids: set[str]
    ) -> tuple[dict[str, str | None], set[str]]:
        """Build mapping from spectrum ID to year."""
        year_map: dict[str, str | None] = {}
        years: set[str] = set()
        for sid in matched_ids:
            yr = self.layout.get_year(sid)
            year_map[sid] = yr
            if yr is not None:
                years.add(yr)
        return year_map, years

    def _create_skeleton(self, years: set[str], all_folders: list[str]) -> None:
        """Create directory structure for the output dataset."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for folder in all_folders:
            if years:
                for yr in years:
                    (self.output_dir / folder / yr).mkdir(parents=True, exist_ok=True)
            else:
                (self.output_dir / folder).mkdir(parents=True, exist_ok=True)

        if years:
            for yr in years:
                (self.output_dir / self.metadata_dir / yr).mkdir(
                    parents=True, exist_ok=True
                )
        else:
            (self.output_dir / self.metadata_dir).mkdir(parents=True, exist_ok=True)

    def _write_metadata(
        self,
        meta: pd.DataFrame,
        matched_ids: set[str],
        year_map: dict[str, str | None],
        years: set[str],
    ) -> None:
        """Write filtered metadata CSV(s) to the output directory."""
        meta_matched = meta[meta["ID"].astype(str).isin(matched_ids)].copy()
        meta_matched = meta_matched.rename(columns={"ID": self.id_column})
        meta_dir = self.metadata_dir
        meta_sfx = self.metadata_suffix

        if years:
            meta_matched["_year"] = (
                meta_matched[self.id_column].astype(str).map(year_map)
            )
            for yr in sorted(years):
                year_meta = meta_matched[meta_matched["_year"] == yr].drop(
                    columns=["_year"]
                )
                year_meta.to_csv(
                    self.output_dir / meta_dir / yr / f"{yr}{meta_sfx}",
                    index=False,
                )
            meta_matched = meta_matched.drop(columns=["_year"])
        else:
            meta_matched.to_csv(
                self.output_dir / meta_dir / f"{self.name}{meta_sfx}",
                index=False,
            )
