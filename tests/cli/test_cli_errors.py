"""Tests for CLI error paths and edge cases."""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from maldiamrkit.cli import _load_extra_handlers, _load_pipeline, app
from maldiamrkit.preprocessing import PreprocessingPipeline

runner = CliRunner()


class TestLoadPipeline:
    """Tests for _load_pipeline helper."""

    def test_none_returns_default(self):
        pipe = _load_pipeline(None)
        assert isinstance(pipe, PreprocessingPipeline)

    def test_json_path(self, tmp_path):
        pipe = PreprocessingPipeline.default()
        json_path = tmp_path / "pipe.json"
        pipe.to_json(str(json_path))
        loaded = _load_pipeline(json_path)
        assert isinstance(loaded, PreprocessingPipeline)
        assert loaded.step_names == pipe.step_names

    def test_yaml_path(self, tmp_path):
        pipe = PreprocessingPipeline.default()
        yaml_path = tmp_path / "pipe.yaml"
        pipe.to_yaml(str(yaml_path))
        loaded = _load_pipeline(yaml_path)
        assert isinstance(loaded, PreprocessingPipeline)


class TestPreprocessErrors:
    """Tests for preprocess command error paths."""

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(
            app,
            [
                "preprocess",
                "--input-dir",
                str(empty_dir),
                "--output",
                str(tmp_path / "out.csv"),
            ],
        )
        assert result.exit_code != 0

    def test_all_spectra_fail(self, tmp_path):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "bad.txt").write_text("not a spectrum")
        result = runner.invoke(
            app,
            [
                "preprocess",
                "--input-dir",
                str(spectra_dir),
                "--output",
                str(tmp_path / "out.csv"),
            ],
        )
        assert result.exit_code != 0


class TestQualityErrors:
    """Tests for quality command error paths."""

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(
            app,
            [
                "quality",
                "--input-dir",
                str(empty_dir),
                "--output",
                str(tmp_path / "out.csv"),
            ],
        )
        assert result.exit_code != 0

    def test_all_spectra_fail(self, tmp_path):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "bad.txt").write_text("not a spectrum")
        result = runner.invoke(
            app,
            [
                "quality",
                "--input-dir",
                str(spectra_dir),
                "--output",
                str(tmp_path / "out.csv"),
            ],
        )
        assert result.exit_code != 0


class TestBuildDriamsErrors:
    """Tests for build command error paths."""

    def test_invalid_extra_handlers_config(self, tmp_path):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        meta_path = tmp_path / "meta.csv"
        meta_path.write_text("ID\nfoo\n")
        handlers_path = tmp_path / "handlers.json"
        handlers_path.write_text('"not a list"')
        result = runner.invoke(
            app,
            [
                "build",
                "--spectra-dir",
                str(spectra_dir),
                "--metadata",
                str(meta_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--extra-handlers",
                str(handlers_path),
            ],
        )
        assert result.exit_code != 0

    def test_builder_value_error(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        meta_path = tmp_path / "meta.csv"
        meta_path.write_text("ID\nfoo\n")
        result = runner.invoke(
            app,
            [
                "build",
                "--spectra-dir",
                str(empty_dir),
                "--metadata",
                str(meta_path),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0


class TestBuildLayoutErrors:
    """Tests for build command layout error paths."""

    def test_invalid_layout_value(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "build",
                "--spectra-dir",
                str(tmp_path),
                "--metadata",
                str(tmp_path / "meta.csv"),
                "--output-dir",
                str(tmp_path / "out"),
                "--layout",
                "invalid",
            ],
        )
        assert result.exit_code != 0


class TestLoadExtraHandlers:
    """Tests for _load_extra_handlers helper."""

    def test_yaml_format(self, tmp_path):
        """Verify YAML config files are loaded correctly."""
        config = [
            {"folder_name": "custom", "kind": "preprocessed"},
        ]
        yaml_path = tmp_path / "handlers.yaml"
        yaml_path.write_text(yaml.dump(config))
        handlers = _load_extra_handlers(yaml_path)
        assert len(handlers) == 1
        assert handlers[0].folder_name == "custom"

    def test_too_large_raises(self, tmp_path):
        """Verify file > 1MB raises BadParameter."""
        import typer

        large_file = tmp_path / "big.json"
        large_file.write_bytes(b"x" * 1_000_001)
        with pytest.raises(typer.BadParameter, match="too large"):
            _load_extra_handlers(large_file)

    def test_not_list_raises(self, tmp_path):
        """Verify non-list config raises BadParameter."""
        import typer

        config_path = tmp_path / "handlers.json"
        config_path.write_text(json.dumps({"key": "value"}))
        with pytest.raises(typer.BadParameter, match="list"):
            _load_extra_handlers(config_path)

    def test_relative_pipeline_path_resolved(self, tmp_path):
        """Verify relative pipeline paths are resolved against config dir."""
        pipe = PreprocessingPipeline.default()
        pipe_path = tmp_path / "pipe.json"
        pipe.to_json(str(pipe_path))
        config = [
            {"folder_name": "custom", "kind": "preprocessed", "pipeline": "pipe.json"},
        ]
        config_path = tmp_path / "handlers.json"
        config_path.write_text(json.dumps(config))
        handlers = _load_extra_handlers(config_path)
        assert handlers[0].pipeline is not None
