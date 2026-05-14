"""Self-describing dataset manifest: ``site_info.json``.

A ``site_info.json`` file lives at the root of every dataset produced by
:class:`maldiamrkit.data.DatasetBuilder` and captures the loader-relevant
settings needed to re-open the dataset without external knowledge.

The format is versioned (``format_version``).  Readers are *lenient*:
they tolerate manifests written by a newer MaldiAMRKit provided all
required fields of the reader's known version are still present.  Hard
breakage only happens when a required field is missing or malformed.

Schema (v1)
-----------
Top level (the "load-time contract", required):

* ``format_version`` (int)
* ``id_column`` (str)
* ``metadata_dir`` (str)
* ``metadata_suffix`` (str, includes the leading ``_`` and the ``.csv``)
* ``spectrum_ext`` (str, includes the leading dot)
* ``spectra_folders`` (list[str])
* ``mz_range`` (list[float] of length 2)
* ``bin_width`` (number)

Optional nested ``build_info`` (provenance, informational only):

* ``maldiamrkit_version`` (str)
* ``created_at`` (ISO 8601 UTC timestamp string)
* ``source_layout`` (class name of the input layout used at build time)
* ``duplicate_strategy`` (str or null)
* ``n_total_spectra`` / ``n_succeeded`` / ``n_failed`` (int)

Downstream readers (notably
:class:`maldiamrkit.data.DRIAMSLayout`) consult the manifest at load
time and use its fields to pre-fill unspecified constructor kwargs.
Explicit kwargs always win.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "site_info.json"
"""On-disk filename for the manifest, written at the dataset root."""

CURRENT_FORMAT_VERSION = 1
"""Manifest schema version produced by *this* MaldiAMRKit release."""

_REQUIRED_V1_KEYS: tuple[str, ...] = (
    "format_version",
    "id_column",
    "metadata_dir",
    "metadata_suffix",
    "spectrum_ext",
    "spectra_folders",
    "mz_range",
    "bin_width",
)


@dataclass
class BuildInfo:
    """Optional provenance block nested under :attr:`SiteInfo.build_info`.

    Informational only; readers may inspect it but are not required to
    interpret any field.  All fields are optional.
    """

    maldiamrkit_version: str | None = None
    created_at: str | None = None
    source_layout: str | None = None
    duplicate_strategy: str | None = None
    n_total_spectra: int | None = None
    n_succeeded: int | None = None
    n_failed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise as a plain dict (omitting ``None`` values for cleanliness)."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SiteInfo:
    """Top-level dataset manifest.

    Parameters
    ----------
    id_column, metadata_dir, metadata_suffix, spectrum_ext : str
        Loader-relevant settings; pre-fill the matching kwargs of
        :class:`DRIAMSLayout`.
    spectra_folders : list[str]
        Sub-directories under the dataset root that contain spectra
        (e.g. ``["raw", "preprocessed", "binned_6000"]``).
    mz_range : tuple[float, float]
        ``(mz_min, mz_max)`` used at build time.
    bin_width : float
        Bin width in Daltons used at build time.
    build_info : BuildInfo, optional
        Optional provenance block.
    format_version : int, default=:data:`CURRENT_FORMAT_VERSION`
        Manifest schema version.
    """

    id_column: str
    metadata_dir: str
    metadata_suffix: str
    spectrum_ext: str
    spectra_folders: list[str]
    mz_range: tuple[float, float]
    bin_width: float
    build_info: BuildInfo | None = None
    format_version: int = CURRENT_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialise as a plain dict, with ``format_version`` first."""
        body: dict[str, Any] = {
            "format_version": int(self.format_version),
            "id_column": self.id_column,
            "metadata_dir": self.metadata_dir,
            "metadata_suffix": self.metadata_suffix,
            "spectrum_ext": self.spectrum_ext,
            "spectra_folders": list(self.spectra_folders),
            "mz_range": [float(self.mz_range[0]), float(self.mz_range[1])],
            "bin_width": float(self.bin_width),
        }
        if self.build_info is not None:
            body["build_info"] = self.build_info.to_dict()
        return body


def write_site_info(dataset_dir: str | Path, site_info: SiteInfo) -> Path:
    """Write a :class:`SiteInfo` to ``<dataset_dir>/site_info.json``.

    Parameters
    ----------
    dataset_dir : str or Path
        Dataset root directory.  Must exist.
    site_info : SiteInfo
        Manifest contents.

    Returns
    -------
    Path
        Path to the written manifest.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot write manifest: dataset directory does not exist: {dataset_dir}"
        )
    path = dataset_dir / MANIFEST_FILENAME
    data = site_info.to_dict()
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    return path


def read_site_info(
    dataset_dir: str | Path,
    *,
    missing_ok: bool = True,
) -> SiteInfo | None:
    """Read ``<dataset_dir>/site_info.json`` if present.

    Parameters
    ----------
    dataset_dir : str or Path
        Dataset root directory.
    missing_ok : bool, default=True
        When ``True`` and the manifest does not exist, return ``None``.
        When ``False``, raise :class:`FileNotFoundError`.

    Returns
    -------
    SiteInfo or None
        Parsed manifest, or ``None`` if absent and ``missing_ok=True``.

    Raises
    ------
    FileNotFoundError
        If the manifest is absent and ``missing_ok=False``.
    ValueError
        If the manifest is malformed, missing a required field, or has
        a non-integer ``format_version``.
    """
    path = Path(dataset_dir) / MANIFEST_FILENAME
    if not path.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(f"{MANIFEST_FILENAME} not found in {dataset_dir}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(raw).__name__}")

    return _site_info_from_dict(raw, source=path)


def _site_info_from_dict(
    raw: dict[str, Any], *, source: Path | None = None
) -> SiteInfo:
    """Parse a dict into :class:`SiteInfo` with lenient version checking."""
    where = f" (in {source})" if source else ""

    fv = raw.get("format_version")
    if fv is None:
        raise ValueError(
            f"site_info{where} is missing required field 'format_version'."
        )
    if not isinstance(fv, int):
        raise ValueError(f"site_info{where} has non-integer format_version={fv!r}.")

    if fv > CURRENT_FORMAT_VERSION:
        warnings.warn(
            f"site_info{where} was written by a newer MaldiAMRKit "
            f"(format_version={fv}; this reader knows v{CURRENT_FORMAT_VERSION}). "
            "Reading what I can; unknown fields will be ignored. "
            "Upgrade `maldiamrkit` if loading misbehaves.",
            UserWarning,
            stacklevel=3,
        )

    missing = [k for k in _REQUIRED_V1_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"site_info{where} is missing required v1 fields: {missing}. "
            "Manifest may be corrupted, hand-edited, or from an unsupported version."
        )

    bi_raw = raw.get("build_info")
    if bi_raw is None:
        build_info = None
    elif isinstance(bi_raw, dict):
        # Only carry through known fields; ignore any future additions silently.
        known = {f.name for f in BuildInfo.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        build_info = BuildInfo(**{k: v for k, v in bi_raw.items() if k in known})
    else:
        raise ValueError(
            f"site_info{where}: 'build_info' must be an object or omitted, "
            f"got {type(bi_raw).__name__}."
        )

    mz_range_raw = raw["mz_range"]
    if not isinstance(mz_range_raw, (list, tuple)) or len(mz_range_raw) != 2:
        raise ValueError(
            f"site_info{where}: 'mz_range' must be a list of two numbers, "
            f"got {mz_range_raw!r}."
        )
    mz_range = (float(mz_range_raw[0]), float(mz_range_raw[1]))

    spectra_folders_raw = raw["spectra_folders"]
    if not isinstance(spectra_folders_raw, list):
        raise ValueError(
            f"site_info{where}: 'spectra_folders' must be a list of strings, "
            f"got {type(spectra_folders_raw).__name__}."
        )

    return SiteInfo(
        id_column=str(raw["id_column"]),
        metadata_dir=str(raw["metadata_dir"]),
        metadata_suffix=str(raw["metadata_suffix"]),
        spectrum_ext=str(raw["spectrum_ext"]),
        spectra_folders=[str(f) for f in spectra_folders_raw],
        mz_range=mz_range,
        bin_width=float(raw["bin_width"]),
        build_info=build_info,
        format_version=fv,
    )


def _current_iso_utc() -> str:
    """Return the current UTC time as an ISO 8601 string with seconds precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "MANIFEST_FILENAME",
    "CURRENT_FORMAT_VERSION",
    "SiteInfo",
    "BuildInfo",
    "read_site_info",
    "write_site_info",
]
