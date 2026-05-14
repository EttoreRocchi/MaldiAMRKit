# Vendored EUCAST clinical breakpoint tables

This directory holds machine-readable EUCAST clinical breakpoint tables in the canonical YAML schema consumed by `maldiamrkit.susceptibility.BreakpointTable`.

## File layout

One YAML file per EUCAST guideline version:

```
eucast_v1.0.yaml
eucast_v2.0.yaml
...
eucast_v16.0.yaml
```

`BreakpointTable.from_version("16.0")` looks up the matching file here. `BreakpointTable.from_latest()` returns the highest-numbered version present. `BreakpointTable.from_year(2026)` searches the `year:` field in the file headers and returns the matching guideline (highest version on a tie).

## YAML schema

```yaml
guideline: EUCAST
version: "16.0"
year: 2026
source: "EUCAST Clinical Breakpoints v16.0 (2026-01-01)"
rows:
  - species: "Klebsiella pneumoniae"
    drug: "Ceftriaxone"
    s_le: 1.0          # MIC ≤ 1.0  →  S
    r_gt: 2.0          # MIC > 2.0  →  R    (otherwise: I)
    atu_low: null      # optional ATU range start (mg/L)
    atu_high: null     # optional ATU range end (mg/L)
```

Categorisation matches EUCAST's published notation verbatim: `S ≤ s_le`, `R > r_gt`. When `s_le == r_gt` there is no `I` zone. `atu_low` / `atu_high` are optional and orthogonal to S/I/R - they flag MICs that sit in the Area of Technical Uncertainty.

## How to populate this directory

EUCAST tables are freely usable with attribution (<https://www.eucast.org/clinical_breakpoints>). The official distribution is an Excel workbook per version. Conversion to this YAML schema is handled by the `eucast_converter/` tooling at the repository root, which is gitignored: it lives only on a maintainer's machine. Workflow:

1. Download the EUCAST Excel for the desired version.
2. Run `python eucast_converter/cli.py path/to/v_16.0_Breakpoint_Tables.xlsx --version 16.0`.
3. Commit the resulting `eucast_v16.0.yaml` to this directory.
4. Bump the package version and release.

The converter targets EUCAST v16.0 layout as its reference; older versions may need shims because EUCAST occasionally re-arranges the workbook.
