"""Command-line interface for MaldiAMRKit.

Provides three subcommands:

- ``preprocess``: Batch preprocess and bin spectra, outputting a CSV feature matrix.
- ``quality``: Compute quality metrics (SNR, TIC, peak count, etc.) for all spectra.
- ``build``: Build a standardised dataset directory from raw spectra and metadata.

Examples
--------
.. code-block:: bash

    maldiamrkit preprocess --input-dir data/ --output processed.csv --bin-width 3
    maldiamrkit quality --input-dir data/ --output report.csv
    maldiamrkit build --spectra-dir data/ --metadata meta.csv --output-dir output/
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .data import BrukerTreeLayout, DatasetBuilder, FlatLayout, ProcessingHandler
from .data.input_layouts import InputLayout
from .io.readers import read_spectrum
from .preprocessing.pipeline import preprocess as preprocess_spectrum
from .preprocessing.preprocessing_pipeline import PreprocessingPipeline
from .preprocessing.quality import SpectrumQuality
from .spectrum import MaldiSpectrum

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="maldiamrkit",
    help="MaldiAMRKit: MALDI-TOF preprocessing toolkit for AMR prediction.",
    add_completion=False,
    rich_markup_mode="rich",
)


def _load_pipeline(path: Path | None) -> PreprocessingPipeline:
    """Load a preprocessing pipeline from file, or return the default."""
    if path is None:
        return PreprocessingPipeline.default()
    if path.suffix in (".yaml", ".yml"):
        return PreprocessingPipeline.from_yaml(path)
    return PreprocessingPipeline.from_json(path)


class BinningMethod(str, Enum):
    """Supported binning methods."""

    uniform = "uniform"
    proportional = "proportional"


class InputLayoutType(str, Enum):
    """Supported input layout types for the build command."""

    flat = "flat"
    bruker = "bruker"


@app.command()
def preprocess(
    input_dir: Annotated[
        Path,
        typer.Option(
            "-i", "--input-dir", help="Directory containing .txt spectrum files."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output CSV file for the feature matrix."),
    ],
    bin_width: Annotated[
        int, typer.Option("-b", "--bin-width", help="Bin width in Daltons.")
    ] = 3,
    method: Annotated[
        BinningMethod, typer.Option(help="Binning method.")
    ] = BinningMethod.uniform,
    pipeline: Annotated[
        Optional[Path],
        typer.Option("-p", "--pipeline", help="JSON/YAML pipeline config."),
    ] = None,
    save_spectra_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to save preprocessed spectra as TXT."),
    ] = None,
) -> None:
    """Batch preprocess and bin spectra to a CSV feature matrix."""
    pipe = _load_pipeline(pipeline)

    spectrum_files = sorted(input_dir.glob("*.txt"))
    if not spectrum_files:
        console.print(f"[red]Error:[/red] No .txt spectrum files found in {input_dir}")
        raise typer.Exit(code=1)

    if save_spectra_dir is not None:
        save_spectra_dir.mkdir(parents=True, exist_ok=True)

    from .preprocessing.binning import bin_spectrum

    mz_min, mz_max = pipe.mz_range
    rows: list[pd.Series] = []
    n_failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing spectra...", total=len(spectrum_files))
        for path in spectrum_files:
            progress.update(task, description=f"Processing {path.name}")
            try:
                raw = read_spectrum(path)
                preprocessed = preprocess_spectrum(raw, pipe)
                if save_spectra_dir is not None:
                    preprocessed.to_csv(
                        save_spectra_dir / f"{path.stem}.txt",
                        sep="\t",
                        index=False,
                    )
                binned, _ = bin_spectrum(
                    preprocessed,
                    mz_min=mz_min,
                    mz_max=mz_max,
                    bin_width=bin_width,
                    method=method.value,
                )
                row = binned.set_index("mass")["intensity"].rename(path.stem)
                rows.append(row)
            except (
                ValueError,
                OSError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
            ) as exc:
                n_failed += 1
                logger.warning("Failed to process %s: %s", path.name, exc)
            progress.advance(task)

    if not rows:
        console.print("[red]Error:[/red] No spectra were successfully processed.")
        raise typer.Exit(code=1)

    feature_matrix = pd.concat(rows, axis=1).T
    feature_matrix.index.name = "ID"
    feature_matrix.to_csv(output)

    # Summary
    console.print()
    table = Table(title="Preprocessing Summary", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Spectra processed", str(len(rows)))
    table.add_row("Failed", str(n_failed))
    table.add_row("Features (bins)", str(feature_matrix.shape[1]))
    table.add_row("Output", str(output))
    if save_spectra_dir is not None:
        table.add_row("Spectra saved to", str(save_spectra_dir))
    console.print(table)


@app.command()
def quality(
    input_dir: Annotated[
        Path,
        typer.Option(
            "-i", "--input-dir", help="Directory containing .txt spectrum files."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output CSV file for the quality report."),
    ],
) -> None:
    """Compute quality metrics (SNR, TIC, peak count, etc.) for all spectra."""
    spectrum_files = sorted(input_dir.glob("*.txt"))
    if not spectrum_files:
        console.print(f"[red]Error:[/red] No .txt spectrum files found in {input_dir}")
        raise typer.Exit(code=1)

    qc = SpectrumQuality()
    reports: list[dict] = []
    n_failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Assessing quality...", total=len(spectrum_files))
        for path in spectrum_files:
            progress.update(task, description=f"Assessing {path.name}")
            try:
                spec = MaldiSpectrum(path)
                report = qc.assess(spec)
                reports.append({"ID": path.stem, **asdict(report)})
            except (
                ValueError,
                OSError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
            ) as exc:
                n_failed += 1
                logger.warning("Failed to assess %s: %s", path.name, exc)
            progress.advance(task)

    if not reports:
        console.print("[red]Error:[/red] No spectra were successfully assessed.")
        raise typer.Exit(code=1)

    report_df = pd.DataFrame(reports)
    report_df.to_csv(output, index=False)

    # Summary
    console.print()
    table = Table(title="Quality Assessment Summary", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Spectra assessed", str(len(reports)))
    table.add_row("Failed", str(n_failed))
    table.add_row("Output", str(output))
    console.print(table)


def _load_extra_handlers(path: Path) -> list[ProcessingHandler]:
    """Load extra handlers from a JSON or YAML config file."""
    file_size = path.stat().st_size
    if file_size > 1_000_000:
        raise typer.BadParameter(f"Config file too large ({file_size} bytes, max 1 MB)")

    if path.suffix in (".yaml", ".yml"):
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        import json

        with open(path) as f:
            data = json.load(f)

    if not isinstance(data, list):
        raise typer.BadParameter(
            f"Extra handlers config must be a YAML/JSON list, got {type(data).__name__}"
        )

    handlers = []
    for entry in data:
        # Resolve relative pipeline paths against the config file directory
        if isinstance(entry.get("pipeline"), str):
            pipeline_path = Path(entry["pipeline"])
            if not pipeline_path.is_absolute():
                entry["pipeline"] = str(path.parent / pipeline_path)
        handlers.append(ProcessingHandler.from_dict(entry))
    return handlers


@app.command("build")
def build(
    spectra_dir: Annotated[
        Path,
        typer.Option(
            "-s", "--spectra-dir", help="Directory containing raw spectrum files."
        ),
    ],
    metadata: Annotated[
        Path, typer.Option("-m", "--metadata", help="Metadata CSV file.")
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o", "--output-dir", help="Output directory for the standardised dataset."
        ),
    ],
    layout: Annotated[
        InputLayoutType, typer.Option("-l", "--layout", help="Input layout type.")
    ] = InputLayoutType.flat,
    name: Annotated[
        Optional[str],
        typer.Option(
            "-n",
            "--name",
            help="Dataset name (for metadata filename). Defaults to output dir name.",
        ),
    ] = None,
    id_column: Annotated[
        Optional[str],
        typer.Option(
            help="Column name for spectrum IDs. Defaults to 'ID' (flat) or 'Identifier' (bruker).",
        ),
    ] = None,
    year_column: Annotated[
        Optional[str],
        typer.Option(
            help="Metadata column to extract year from for year-based subfolders."
        ),
    ] = None,
    bin_width: Annotated[
        int, typer.Option("-b", "--bin-width", help="Bin width in Daltons.")
    ] = 3,
    pipeline: Annotated[
        Optional[Path],
        typer.Option("-p", "--pipeline", help="JSON/YAML pipeline config."),
    ] = None,
    extra_handlers: Annotated[
        Optional[Path],
        typer.Option(help="JSON/YAML config file defining extra processing handlers."),
    ] = None,
    n_jobs: Annotated[
        int, typer.Option("-j", "--n-jobs", help="Parallel jobs (-1 = all cores).")
    ] = -1,
    # Bruker-specific options
    path_column: Annotated[
        str,
        typer.Option(
            help="Metadata column with path to Bruker directory (bruker layout only)."
        ),
    ] = "Path",
    target_position_column: Annotated[
        str,
        typer.Option(
            help="Metadata column for plate target position (bruker layout only)."
        ),
    ] = "target_position",
    deduplicate: Annotated[
        bool,
        typer.Option(help="Keep one spectrum per identifier (bruker layout only)."),
    ] = True,
    validate: Annotated[
        bool,
        typer.Option(
            help="Skip empty spectra and warn on duplicates (bruker layout only)."
        ),
    ] = True,
) -> None:
    """Build a standardised dataset directory from raw spectra and metadata."""
    pipe = _load_pipeline(pipeline)

    # Resolve layout-aware id_column default
    resolved_id = id_column or (
        "Identifier" if layout == InputLayoutType.bruker else "ID"
    )

    # Parse extra handlers
    handlers = None
    if extra_handlers is not None:
        try:
            handlers = _load_extra_handlers(Path(extra_handlers))
        except Exception as exc:
            console.print(f"[red]Error loading extra handlers:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Building dataset...", total=None)
        try:
            input_layout: InputLayout
            if layout == InputLayoutType.flat:
                input_layout = FlatLayout(
                    spectra_dir,
                    metadata,
                    id_column=resolved_id,
                    year_column=year_column,
                )
            else:
                input_layout = BrukerTreeLayout(
                    spectra_dir,
                    metadata,
                    id_column=resolved_id,
                    year_column=year_column or "Year",
                    path_column=path_column,
                    target_position_column=target_position_column,
                    deduplicate=deduplicate,
                    validate=validate,
                )
            builder = DatasetBuilder(
                input_layout,
                output_dir,
                name=name,
                id_column=resolved_id,
                pipeline=pipe,
                bin_width=bin_width,
                extra_handlers=handlers,
                n_jobs=n_jobs,
            )
            report = builder.build()
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    # Summary
    console.print()
    table = Table(title="Build Summary", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Spectra processed", str(report.succeeded))
    table.add_row("Failed", str(report.failed))
    table.add_row("Folders created", ", ".join(report.folders_created))
    table.add_row("Output", str(report.output_dir))
    console.print(table)

    if report.failed > 0:
        console.print(
            f"\n[yellow]Warning:[/yellow] {report.failed} spectra failed. "
            f"Check logs for details."
        )


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
