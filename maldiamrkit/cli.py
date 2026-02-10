"""Command-line interface for MaldiAMRKit.

Provides two subcommands:

- ``preprocess``: Batch preprocess and bin spectra, outputting a CSV feature matrix.
- ``quality``: Compute quality metrics (SNR, TIC, peak count, etc.) for all spectra.

Examples
--------
.. code-block:: bash

    maldiamrkit preprocess --input-dir data/ --output processed.csv --bin-width 3
    maldiamrkit quality --input-dir data/ --output report.csv
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


class BinningMethod(str, Enum):
    """Supported binning methods."""

    uniform = "uniform"
    logarithmic = "logarithmic"


@app.command()
def preprocess(
    input_dir: Annotated[
        Path, typer.Option(help="Directory containing .txt spectrum files.")
    ],
    output: Annotated[
        Path, typer.Option(help="Output CSV file for the feature matrix.")
    ],
    bin_width: Annotated[int, typer.Option(help="Bin width in Daltons.")] = 3,
    method: Annotated[
        BinningMethod, typer.Option(help="Binning method.")
    ] = BinningMethod.uniform,
    pipeline: Annotated[
        Optional[Path], typer.Option(help="JSON/YAML pipeline config.")
    ] = None,
    save_spectra_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to save preprocessed spectra as TXT."),
    ] = None,
) -> None:
    """Batch preprocess and bin spectra to a CSV feature matrix."""
    if pipeline is not None:
        pipeline_path = Path(pipeline)
        if pipeline_path.suffix in (".yaml", ".yml"):
            pipe = PreprocessingPipeline.from_yaml(pipeline_path)
        else:
            pipe = PreprocessingPipeline.from_json(pipeline_path)
    else:
        pipe = PreprocessingPipeline.default()

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
            except Exception as exc:
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
        Path, typer.Option(help="Directory containing .txt spectrum files.")
    ],
    output: Annotated[
        Path, typer.Option(help="Output CSV file for the quality report.")
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
            except Exception as exc:
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


if __name__ == "__main__":
    app()
