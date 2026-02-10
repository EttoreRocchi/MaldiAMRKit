"""Unit tests for pipeline serialization (JSON/YAML)."""

from pathlib import Path

import pandas as pd

from maldiamrkit.preprocessing import (
    ClipNegatives,
    LogTransform,
    MzTrimmer,
    PreprocessingPipeline,
    SavitzkyGolaySmooth,
    SNIPBaseline,
    SqrtTransform,
    TICNormalizer,
)


class TestDictRoundtrip:
    """Tests for to_dict / from_dict."""

    def test_default_pipeline_roundtrip(self):
        pipe = PreprocessingPipeline.default()
        d = pipe.to_dict()
        pipe2 = PreprocessingPipeline.from_dict(d)

        assert pipe.step_names == pipe2.step_names
        assert pipe.mz_range == pipe2.mz_range

    def test_custom_pipeline_roundtrip(self):
        pipe = PreprocessingPipeline(
            [
                ("clip", ClipNegatives()),
                ("log", LogTransform()),
                ("smooth", SavitzkyGolaySmooth(window_length=15, polyorder=3)),
                ("baseline", SNIPBaseline(half_window=30)),
                ("trim", MzTrimmer(mz_min=3000, mz_max=15000)),
                ("norm", TICNormalizer()),
            ]
        )
        d = pipe.to_dict()
        pipe2 = PreprocessingPipeline.from_dict(d)

        assert pipe2.step_names == ["clip", "log", "smooth", "baseline", "trim", "norm"]
        assert pipe2.mz_range == (3000, 15000)

        # Verify transformer parameters survived
        smooth = pipe2.get_step("smooth")
        assert isinstance(smooth, SavitzkyGolaySmooth)
        assert smooth.window_length == 15
        assert smooth.polyorder == 3

    def test_dict_contains_expected_keys(self):
        pipe = PreprocessingPipeline.default()
        d = pipe.to_dict()

        assert "steps" in d
        for step in d["steps"]:
            assert "step_name" in step
            assert "name" in step

    def test_roundtrip_produces_same_output(self, synthetic_spectrum: pd.DataFrame):
        pipe = PreprocessingPipeline.default()
        result1 = pipe(synthetic_spectrum.copy())

        pipe2 = PreprocessingPipeline.from_dict(pipe.to_dict())
        result2 = pipe2(synthetic_spectrum.copy())

        pd.testing.assert_frame_equal(result1, result2)


class TestJsonSerialization:
    """Tests for to_json / from_json."""

    def test_json_roundtrip(self, tmp_path: Path):
        path = tmp_path / "test_pipeline.json"
        pipe = PreprocessingPipeline.default()
        pipe.to_json(path)

        pipe2 = PreprocessingPipeline.from_json(path)

        assert pipe.step_names == pipe2.step_names
        assert pipe.mz_range == pipe2.mz_range

    def test_json_file_is_readable(self, tmp_path: Path):
        import json

        path = tmp_path / "test_pipeline_readable.json"
        pipe = PreprocessingPipeline.default()
        pipe.to_json(path)

        with open(path) as f:
            d = json.load(f)
        assert "steps" in d

    def test_json_custom_pipeline(self, tmp_path: Path):
        path = tmp_path / "test_custom.json"
        pipe = PreprocessingPipeline(
            [
                ("trim", MzTrimmer(mz_min=3000, mz_max=15000)),
                ("norm", TICNormalizer()),
            ]
        )
        pipe.to_json(path)
        pipe2 = PreprocessingPipeline.from_json(path)

        assert pipe2.mz_range == (3000, 15000)
        assert len(pipe2) == 2

    def test_json_produces_same_output(
        self, tmp_path: Path, synthetic_spectrum: pd.DataFrame
    ):
        path = tmp_path / "test_output.json"
        pipe = PreprocessingPipeline.default()
        pipe.to_json(path)

        pipe2 = PreprocessingPipeline.from_json(path)
        result1 = pipe(synthetic_spectrum.copy())
        result2 = pipe2(synthetic_spectrum.copy())

        pd.testing.assert_frame_equal(result1, result2)


class TestYamlSerialization:
    """Tests for to_yaml / from_yaml."""

    def test_yaml_roundtrip(self, tmp_path: Path):
        path = tmp_path / "test_pipeline.yaml"
        pipe = PreprocessingPipeline.default()
        pipe.to_yaml(path)

        pipe2 = PreprocessingPipeline.from_yaml(path)

        assert pipe.step_names == pipe2.step_names
        assert pipe.mz_range == pipe2.mz_range

    def test_yaml_file_is_readable(self, tmp_path: Path):
        import yaml

        path = tmp_path / "test_pipeline_readable.yaml"
        pipe = PreprocessingPipeline.default()
        pipe.to_yaml(path)

        with open(path) as f:
            d = yaml.safe_load(f)
        assert "steps" in d

    def test_yaml_custom_pipeline(self, tmp_path: Path):
        path = tmp_path / "test_custom.yaml"
        pipe = PreprocessingPipeline(
            [
                ("clip", ClipNegatives()),
                ("sqrt", SqrtTransform()),
                ("trim", MzTrimmer(mz_min=4000, mz_max=18000)),
            ]
        )
        pipe.to_yaml(path)
        pipe2 = PreprocessingPipeline.from_yaml(path)

        assert pipe2.mz_range == (4000, 18000)
        assert pipe2.step_names == ["clip", "sqrt", "trim"]

    def test_yaml_produces_same_output(
        self, tmp_path: Path, synthetic_spectrum: pd.DataFrame
    ):
        path = tmp_path / "test_output.yaml"
        pipe = PreprocessingPipeline.default()
        pipe.to_yaml(path)

        pipe2 = PreprocessingPipeline.from_yaml(path)
        result1 = pipe(synthetic_spectrum.copy())
        result2 = pipe2(synthetic_spectrum.copy())

        pd.testing.assert_frame_equal(result1, result2)
