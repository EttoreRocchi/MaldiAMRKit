"""Clinical breakpoint tables for MIC interpretation.

A :class:`BreakpointTable` maps each ``(species, drug)`` pair to ``S ≤ s_le``
and ``R > r_gt`` thresholds, optionally with an Area of Technical Uncertainty
(ATU) range. Categorisation:

- ``mic ≤ s_le``  →  ``"S"`` (Susceptible, standard dosing)
- ``mic > r_gt``  →  ``"R"`` (Resistant)
- otherwise       →  ``"I"`` (Susceptible, increased exposure -- modern EUCAST)

The ATU flag is *orthogonal* to S/I/R: it marks MICs that fall in a zone where
assay variability can flip the call. Treat it as an "investigate further"
warning, not a third clinical category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

_VERSION_FILENAME_RE = re.compile(r"^eucast_v(?P<version>[\d.]+)\.yaml$")
_REQUIRED_ROW_FIELDS = ("species", "drug", "s_le", "r_gt")
_OPTIONAL_ROW_FIELDS = ("atu_low", "atu_high")


@dataclass(frozen=True)
class BreakpointResult:
    """Result of applying a clinical breakpoint to a single MIC value.

    Attributes
    ----------
    category : {"S", "I", "R"} or None
        Clinical category. ``"S"`` (Susceptible, standard dosing),
        ``"I"`` (Susceptible, increased exposure -- modern EUCAST),
        or ``"R"`` (Resistant). ``None`` when the lookup failed
        (no row for this ``(species, drug)``, or MIC is NaN).
    atu : bool
        True when the MIC value falls in the species/drug ATU range.
        Orthogonal to ``category`` -- not a third clinical category.
    source : str or None
        Provenance string, e.g. ``"EUCAST v16.0"``. ``None`` when the
        lookup failed.
    """

    category: str | None
    atu: bool
    source: str | None


class BreakpointTable:
    """Clinical breakpoint table for MIC interpretation.

    Holds a set of ``(species, drug) → (s_le, r_gt, [atu_low, atu_high])``
    rows from a single guideline release (e.g. EUCAST v16.0). Use
    :meth:`apply` for single MICs and :meth:`apply_batch` for arrays;
    :class:`~maldiamrkit.susceptibility.MICEncoder` consumes the batch API.

    Parameters
    ----------
    rows : pd.DataFrame
        DataFrame with at least the columns ``species``, ``drug``, ``s_le``,
        ``r_gt``. Optional columns: ``atu_low``, ``atu_high``.
    guideline : str, default="EUCAST"
        e.g. ``"EUCAST"``.
    version : str, default=""
        Guideline version, e.g. ``"16.0"``.
    year : int or None, default=None
        Calendar year the guideline was published.
    source : str or None, default=None
        Free-text provenance, e.g. ``"EUCAST Clinical Breakpoints v16.0 (2026-01-01)"``.

    Raises
    ------
    ValueError
        If required columns are missing, threshold types are not numeric,
        or any row violates ``s_le ≤ r_gt``.

    Notes
    -----
    EUCAST's literal table format is preserved: ``s_le`` is the largest MIC
    classified as ``S`` and ``r_gt`` is the largest MIC *not* classified as
    ``R``. When ``s_le == r_gt`` there is no ``I`` zone.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        guideline: str = "EUCAST",
        version: str = "",
        year: int | None = None,
        source: str | None = None,
    ) -> None:
        self._rows = self._validate_rows(rows)
        self.guideline = guideline
        self.version = version
        self.year = year
        self.source = source or self._default_source()
        self._lookup: dict[tuple[str, str], int] = {
            (str(r.species).strip().lower(), str(r.drug).strip().lower()): idx
            for idx, r in self._rows.iterrows()
        }

    def __repr__(self) -> str:
        n = len(self._rows)
        return (
            f"BreakpointTable({self.guideline} v{self.version}, "
            f"{n} row{'s' if n != 1 else ''})"
        )

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> pd.DataFrame:
        """Return a copy of the underlying breakpoint rows."""
        return self._rows.copy()

    def species(self) -> list[str]:
        """List unique species present in the table."""
        return sorted(self._rows["species"].unique().tolist())

    def drugs(self) -> list[str]:
        """List unique drugs present in the table."""
        return sorted(self._rows["drug"].unique().tolist())

    def apply(self, species: str, drug: str, mic: float | None) -> BreakpointResult:
        """Categorise a single MIC value against the table.

        Parameters
        ----------
        species : str
            Bacterial species, e.g. ``"Klebsiella pneumoniae"``. Matched
            case-insensitively against the table.
        drug : str
            Antibiotic name. Matched case-insensitively.
        mic : float or None
            MIC value in mg/L (linear scale, not ``log2``). ``None`` /
            ``NaN`` returns a result with ``category=None``.

        Returns
        -------
        BreakpointResult
            See :class:`BreakpointResult`.
        """
        key = (str(species).strip().lower(), str(drug).strip().lower())
        idx = self._lookup.get(key)
        if idx is None:
            return BreakpointResult(category=None, atu=False, source=None)
        if mic is None or (isinstance(mic, float) and np.isnan(mic)):
            return BreakpointResult(category=None, atu=False, source=self.source)
        row = self._rows.loc[idx]
        return BreakpointResult(
            category=self._categorise(float(mic), float(row.s_le), float(row.r_gt)),
            atu=self._in_atu(float(mic), row.atu_low, row.atu_high),
            source=self.source,
        )

    def apply_batch(
        self,
        species: str | Sequence[str] | np.ndarray | pd.Series,
        drug: str | Sequence[str] | np.ndarray | pd.Series,
        mic: Sequence[float] | np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Categorise an array of MIC values.

        ``species`` and ``drug`` may be scalars (broadcast to all rows) or
        arrays of the same length as ``mic``.

        Parameters
        ----------
        species : str or array-like
            Species per sample, or a single species applied to all.
        drug : str or array-like
            Drug per sample, or a single drug applied to all.
        mic : array-like
            MIC values in mg/L (linear scale).

        Returns
        -------
        pd.DataFrame
            Columns: ``category`` (object, ``"S"``/``"I"``/``"R"``/NA),
            ``atu`` (bool), ``source`` (object, possibly NA for unmatched rows).
        """
        mic_arr = pd.Series(mic).astype(float).to_numpy()
        n = len(mic_arr)
        species_arr = _broadcast(species, n, "species")
        drug_arr = _broadcast(drug, n, "drug")

        categories = np.full(n, None, dtype=object)
        atu_flags = np.zeros(n, dtype=bool)
        sources = np.full(n, None, dtype=object)
        for i in range(n):
            res = self.apply(species_arr[i], drug_arr[i], mic_arr[i])
            categories[i] = res.category
            atu_flags[i] = res.atu
            sources[i] = res.source
        return pd.DataFrame(
            {"category": categories, "atu": atu_flags, "source": sources}
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> BreakpointTable:
        """Load a breakpoint table from a YAML file.

        The YAML must have keys ``guideline``, ``version``, optional ``year``
        and ``source``, and a ``rows`` list whose entries carry
        ``species, drug, s_le, r_gt`` and optionally ``atu_low, atu_high``.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return cls._from_payload(payload)

    @classmethod
    def from_version(cls, version: str) -> BreakpointTable:
        """Load a bundled EUCAST table by version string, e.g. ``"16.0"``."""
        version = str(version).strip().lstrip("vV")
        available = cls.list_available()
        if version not in available:
            raise FileNotFoundError(
                f"No bundled EUCAST v{version} table found. "
                f"Available versions: {available or '[]'}. "
                f"Generate vendored YAMLs by running the gitignored "
                f"eucast_converter/ tooling on the official EUCAST workbook."
            )
        return cls._load_bundled(f"eucast_v{version}.yaml")

    @classmethod
    def from_year(cls, year: int) -> BreakpointTable:
        """Load a bundled EUCAST table by calendar year of publication.

        EUCAST publishes annually but the version-to-year mapping isn't a
        clean function (mid-year dot releases exist). When several bundled
        versions match the same year, the highest version is returned.
        """
        candidates: list[tuple[str, BreakpointTable]] = []
        for v in cls.list_available():
            table = cls._load_bundled(f"eucast_v{v}.yaml")
            if table.year == year:
                candidates.append((v, table))
        if not candidates:
            raise FileNotFoundError(
                f"No bundled EUCAST table found for year {year}. "
                f"Available years: {sorted({t.year for _, t in cls._iter_bundled() if t.year})}."
            )
        candidates.sort(key=lambda item: _version_tuple(item[0]), reverse=True)
        return candidates[0][1]

    @classmethod
    def from_latest(cls) -> BreakpointTable:
        """Load the highest-numbered bundled EUCAST table."""
        available = cls.list_available()
        if not available:
            raise FileNotFoundError(
                "No bundled EUCAST tables shipped with this install. "
                "Generate vendored YAMLs by running the gitignored "
                "eucast_converter/ tooling on the official EUCAST workbook."
            )
        latest = max(available, key=_version_tuple)
        return cls._load_bundled(f"eucast_v{latest}.yaml")

    @classmethod
    def list_available(cls) -> list[str]:
        """List bundled EUCAST version strings, sorted numerically."""
        versions: list[str] = []
        try:
            with resources.as_file(_BUNDLED_EUCAST_DIR) as eucast_dir:
                for entry in eucast_dir.iterdir():
                    m = _VERSION_FILENAME_RE.match(entry.name)
                    if m:
                        versions.append(m.group("version"))
        except (FileNotFoundError, ModuleNotFoundError):
            pass
        return sorted(versions, key=_version_tuple)

    @classmethod
    def _iter_bundled(cls) -> Iterable[tuple[str, BreakpointTable]]:
        for v in cls.list_available():
            yield v, cls._load_bundled(f"eucast_v{v}.yaml")

    @classmethod
    def _load_bundled(cls, filename: str) -> BreakpointTable:
        with resources.as_file(_BUNDLED_EUCAST_DIR / filename) as path:
            return cls.from_yaml(path)

    @classmethod
    def _from_payload(cls, payload: dict) -> BreakpointTable:
        rows_payload = payload.get("rows") or []
        if not rows_payload:
            raise ValueError("YAML payload contains no 'rows' entries.")
        rows = pd.DataFrame(rows_payload)
        for field in _OPTIONAL_ROW_FIELDS:
            if field not in rows.columns:
                rows[field] = np.nan
        return cls(
            rows=rows,
            guideline=payload.get("guideline", "EUCAST"),
            version=str(payload.get("version", "")),
            year=payload.get("year"),
            source=payload.get("source"),
        )

    @staticmethod
    def _validate_rows(rows: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in _REQUIRED_ROW_FIELDS if c not in rows.columns]
        if missing:
            raise ValueError(
                f"Breakpoint rows missing required columns: {missing}. "
                f"Expected: {list(_REQUIRED_ROW_FIELDS)}."
            )
        out = rows.copy()
        for field in _OPTIONAL_ROW_FIELDS:
            if field not in out.columns:
                out[field] = np.nan
        out["s_le"] = pd.to_numeric(out["s_le"], errors="raise")
        out["r_gt"] = pd.to_numeric(out["r_gt"], errors="raise")
        out["atu_low"] = pd.to_numeric(out["atu_low"], errors="coerce")
        out["atu_high"] = pd.to_numeric(out["atu_high"], errors="coerce")
        out["species"] = out["species"].astype(str).str.strip()
        out["drug"] = out["drug"].astype(str).str.strip()
        bad = out[out["s_le"] > out["r_gt"]]
        if not bad.empty:
            sample = bad.head(3).to_dict(orient="records")
            raise ValueError(
                f"Found {len(bad)} row(s) with s_le > r_gt (invalid). "
                f"First offenders: {sample}"
            )
        out = out.reset_index(drop=True)
        return out[["species", "drug", "s_le", "r_gt", "atu_low", "atu_high"]]

    def _default_source(self) -> str:
        return f"{self.guideline} v{self.version}" if self.version else self.guideline

    @staticmethod
    def _categorise(mic: float, s_le: float, r_gt: float) -> str:
        if mic <= s_le:
            return "S"
        if mic > r_gt:
            return "R"
        return "I"

    @staticmethod
    def _in_atu(mic: float, atu_low: float, atu_high: float) -> bool:
        if pd.isna(atu_low):
            return False
        if pd.isna(atu_high):
            return bool(mic == atu_low)
        return bool((mic >= atu_low) and (mic <= atu_high))


def _broadcast(value, n: int, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)) or np.isscalar(value):
        return np.full(n, value, dtype=object)
    arr = np.asarray(value, dtype=object)
    if arr.shape[0] != n:
        raise ValueError(
            f"{name!r} length {arr.shape[0]} does not match MIC array length {n}."
        )
    return arr


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


_BUNDLED_EUCAST_DIR = resources.files("maldiamrkit") / "data" / "breakpoints" / "eucast"
