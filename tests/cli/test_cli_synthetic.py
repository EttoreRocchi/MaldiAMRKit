"""Self-contained CLI tests using synthetic spectra (no DATA_DIR dependency)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from maldiamrkit.cli import _load_extra_handlers, _load_pipeline, app
from maldiamrkit.preprocessing import PreprocessingPipeline
from tests.conftest import _generate_synthetic_spectrum

runner = CliRunner()


def _write_synthetic_spectra(tmp_path: Path, n: int = 3) -> Path:
    """Write *n* synthetic spectrum .txt files to a temp directory."""
    spectra_dir = tmp_path / "spectra"
    spectra_dir.mkdir()
    for i in range(n):
        df = _generate_synthetic_spectrum(random_state=42 + i)
        np.savetxt(
            spectra_dir / f"s{i}.txt",
            df[["mass", "intensity"]].values,
            header="mass intensity",
            comments="# ",
            fmt="%.6f",
        )
    return spectra_dir


class TestPreprocessSynthetic:
    """Preprocess command tests with synthetic data."""

    def test_end_to_end_synthetic(self, tmp_path: Path):
        """Verify preprocess produces a valid CSV from synthetic spectra."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        output = tmp_path / "matrix.csv"
        result = runner.invoke(
            app,
            ["preprocess", "-i", str(spectra_dir), "-o", str(output), "-b", "3"],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_output_csv_shape_and_types(self, tmp_path: Path):
        """Verify output CSV has numeric columns and non-negative values."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        output = tmp_path / "matrix.csv"
        runner.invoke(
            app,
            ["preprocess", "-i", str(spectra_dir), "-o", str(output), "-b", "3"],
        )
        df = pd.read_csv(output, index_col=0)
        assert df.shape[0] == 3
        assert df.shape[1] > 0
        assert df.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all()
        assert (df.values >= 0).all()
        assert df.index.name == "ID"

    def test_binning_method_proportional(self, tmp_path: Path):
        """Verify --method proportional runs without error."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        output = tmp_path / "proportional.csv"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "-i",
                str(spectra_dir),
                "-o",
                str(output),
                "--method",
                "proportional",
            ],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_pipeline_json_config(self, tmp_path: Path):
        """Verify preprocess accepts a --pipeline JSON config."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        pipe_path = tmp_path / "pipe.json"
        PreprocessingPipeline.default().to_json(pipe_path)
        output = tmp_path / "from_json.csv"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "-i",
                str(spectra_dir),
                "-o",
                str(output),
                "-p",
                str(pipe_path),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_pipeline_yaml_config(self, tmp_path: Path):
        """Verify preprocess accepts a --pipeline YAML config."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        pipe_path = tmp_path / "pipe.yaml"
        PreprocessingPipeline.default().to_yaml(pipe_path)
        output = tmp_path / "from_yaml.csv"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "-i",
                str(spectra_dir),
                "-o",
                str(output),
                "-p",
                str(pipe_path),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_save_spectra_dir_creates_files(self, tmp_path: Path):
        """Verify --save-spectra-dir writes preprocessed spectra."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        output = tmp_path / "matrix.csv"
        save_dir = tmp_path / "preprocessed"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "-i",
                str(spectra_dir),
                "-o",
                str(output),
                "--save-spectra-dir",
                str(save_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert save_dir.exists()
        txt_files = list(save_dir.glob("*.txt"))
        assert len(txt_files) == 3

    def test_corrupt_file_counted_as_failed(self, tmp_path: Path):
        """Verify that a corrupt file is silently skipped."""
        spectra_dir = _write_synthetic_spectra(tmp_path, n=2)
        # Write an empty (corrupt) file
        (spectra_dir / "bad.txt").write_text("")
        output = tmp_path / "matrix.csv"
        result = runner.invoke(
            app,
            ["preprocess", "-i", str(spectra_dir), "-o", str(output)],
        )
        # Should succeed with the 2 valid files
        assert result.exit_code == 0, result.output
        df = pd.read_csv(output, index_col=0)
        assert df.shape[0] == 2


class TestQualitySynthetic:
    """Quality command tests with synthetic data."""

    def test_end_to_end_synthetic(self, tmp_path: Path):
        """Verify quality produces a CSV report from synthetic spectra."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        output = tmp_path / "quality.csv"
        result = runner.invoke(
            app,
            ["quality", "-i", str(spectra_dir), "-o", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_report_columns_present(self, tmp_path: Path):
        """Verify all expected quality columns are present."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        output = tmp_path / "quality.csv"
        runner.invoke(app, ["quality", "-i", str(spectra_dir), "-o", str(output)])
        df = pd.read_csv(output)
        for col in [
            "ID",
            "snr",
            "total_ion_count",
            "peak_count",
            "baseline_fraction",
            "noise_level",
            "dynamic_range",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_report_values_sane(self, tmp_path: Path):
        """Verify quality metric values are within expected ranges."""
        spectra_dir = _write_synthetic_spectra(tmp_path)
        output = tmp_path / "quality.csv"
        runner.invoke(app, ["quality", "-i", str(spectra_dir), "-o", str(output)])
        df = pd.read_csv(output)
        assert (df["snr"] > 0).all()
        assert (df["baseline_fraction"] >= 0).all()
        assert (df["baseline_fraction"] <= 1).all()
        assert (df["peak_count"] >= 0).all()

    def test_corrupt_file_counted_as_failed(self, tmp_path: Path):
        """Verify corrupt file is skipped without crashing."""
        spectra_dir = _write_synthetic_spectra(tmp_path, n=2)
        (spectra_dir / "bad.txt").write_text("")
        output = tmp_path / "quality.csv"
        result = runner.invoke(
            app, ["quality", "-i", str(spectra_dir), "-o", str(output)]
        )
        assert result.exit_code == 0, result.output
        df = pd.read_csv(output)
        assert len(df) == 2


class TestLoadPipeline:
    """Tests for _load_pipeline helper."""

    def test_none_returns_default(self):
        """Verify _load_pipeline(None) returns the default pipeline."""
        pipe = _load_pipeline(None)
        assert isinstance(pipe, PreprocessingPipeline)
        assert len(pipe.step_names) > 0

    def test_json_path_loads(self, tmp_path: Path):
        """Verify _load_pipeline loads from JSON."""
        pipe_path = tmp_path / "pipe.json"
        PreprocessingPipeline.default().to_json(pipe_path)
        pipe = _load_pipeline(pipe_path)
        assert isinstance(pipe, PreprocessingPipeline)

    def test_yaml_path_loads(self, tmp_path: Path):
        """Verify _load_pipeline loads from YAML."""
        pipe_path = tmp_path / "pipe.yaml"
        PreprocessingPipeline.default().to_yaml(pipe_path)
        pipe = _load_pipeline(pipe_path)
        assert isinstance(pipe, PreprocessingPipeline)


class TestLoadExtraHandlers:
    """Tests for _load_extra_handlers helper."""

    def test_valid_json_list(self, tmp_path: Path):
        """Verify valid JSON list of handler dicts is parsed."""
        config = [{"folder_name": "binned_3000", "kind": "binned", "bin_width": 6}]
        config_path = tmp_path / "handlers.json"
        config_path.write_text(json.dumps(config))
        handlers = _load_extra_handlers(config_path)
        assert len(handlers) == 1
        assert handlers[0].folder_name == "binned_3000"

    def test_valid_yaml_list(self, tmp_path: Path):
        """Verify valid YAML list of handler dicts is parsed."""
        config = [{"folder_name": "binned_3000", "kind": "binned", "bin_width": 6}]
        config_path = tmp_path / "handlers.yaml"
        config_path.write_text(yaml.dump(config))
        handlers = _load_extra_handlers(config_path)
        assert len(handlers) == 1

    def test_non_list_raises(self, tmp_path: Path):
        """Verify non-list config raises BadParameter."""
        import typer

        config_path = tmp_path / "handlers.json"
        config_path.write_text(json.dumps({"folder_name": "x"}))
        with pytest.raises(typer.BadParameter, match="list"):
            _load_extra_handlers(config_path)

    def test_oversized_file_raises(self, tmp_path: Path):
        """Verify file >1MB raises BadParameter."""
        import typer

        config_path = tmp_path / "huge.json"
        config_path.write_text("[" + "0," * 500_000 + "0]")
        with pytest.raises(typer.BadParameter, match="too large"):
            _load_extra_handlers(config_path)

    def test_relative_pipeline_path_resolved(self, tmp_path: Path):
        """Verify relative pipeline path is resolved against config dir."""
        pipe_path = tmp_path / "pipe.json"
        PreprocessingPipeline.default().to_json(pipe_path)

        config = [
            {
                "folder_name": "custom",
                "kind": "preprocessed",
                "pipeline": "pipe.json",
                "bin_width": 3,
            }
        ]
        config_path = tmp_path / "handlers.json"
        config_path.write_text(json.dumps(config))
        handlers = _load_extra_handlers(config_path)
        assert handlers[0].pipeline is not None
