"""Tests for the public transformer registry and pipeline round-tripping."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from maldiamrkit.preprocessing import (
    PreprocessingPipeline,
    list_transformers,
    register_transformer,
    unregister_transformer,
)
from maldiamrkit.preprocessing.transformers import (
    _DEFAULT_TRANSFORMERS,
    _TRANSFORMER_REGISTRY,
)
from tests.conftest import make_registry_guard

_BUILTINS = sorted(_DEFAULT_TRANSFORMERS)

clean_registry = make_registry_guard(_TRANSFORMER_REGISTRY)


class Scale:
    """Custom transformer used across these tests."""

    def __init__(self, factor: float = 2.0):
        self.factor = factor

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["intensity"] = df["intensity"] * self.factor
        return df

    def to_dict(self) -> dict:
        return {"name": "Scale", "factor": self.factor}


class MissingToDict:
    """Callable but not serialisable."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class MissingCall:
    """Serialisable but not callable."""

    def to_dict(self) -> dict:
        return {"name": "MissingCall"}


class WrongName:
    """``to_dict`` declares a name that will not be registered."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def to_dict(self) -> dict:
        return {"name": "SomethingElse"}


class NoNameKey:
    """``to_dict`` omits the ``name`` key entirely."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def to_dict(self) -> dict:
        return {"factor": 1.0}


@pytest.fixture
def spectrum():
    return pd.DataFrame({"mass": [2000.0, 2001.0], "intensity": [1.0, 2.0]})


class TestRegister:
    def test_appears_in_listing(self, clean_registry):
        assert "Scale" not in list_transformers()
        register_transformer("Scale", Scale)
        assert "Scale" in list_transformers()

    def test_listing_is_sorted_and_includes_builtins(self):
        listed = list_transformers()
        assert listed == sorted(listed)
        assert set(_BUILTINS) <= set(listed)

    def test_rejects_instance(self, clean_registry):
        with pytest.raises(TypeError, match="must be a class"):
            register_transformer("Scale", Scale())

    def test_rejects_class_without_to_dict(self, clean_registry):
        with pytest.raises(TypeError, match="to_dict"):
            register_transformer("MissingToDict", MissingToDict)

    def test_rejects_class_without_call(self, clean_registry):
        with pytest.raises(TypeError, match=r"missing __call__\(\)"):
            register_transformer("MissingCall", MissingCall)

    def test_builtin_protected_without_override(self, clean_registry):
        with pytest.raises(ValueError, match="built-in transformer"):
            register_transformer("SNIPBaseline", Scale)
        assert _TRANSFORMER_REGISTRY["SNIPBaseline"] is not Scale

    def test_builtin_replaceable_with_override(self, clean_registry):
        register_transformer("SNIPBaseline", Scale, override=True)
        assert _TRANSFORMER_REGISTRY["SNIPBaseline"] is Scale

    def test_custom_name_can_be_replaced_silently(self, clean_registry):
        register_transformer("Scale", Scale)
        register_transformer("Scale", WrongName)
        assert _TRANSFORMER_REGISTRY["Scale"] is WrongName


class TestUnregister:
    def test_removes_custom_transformer(self, clean_registry):
        register_transformer("Scale", Scale)
        unregister_transformer("Scale")
        assert "Scale" not in list_transformers()

    def test_unknown_name_raises_key_error(self, clean_registry):
        with pytest.raises(KeyError, match="No transformer named"):
            unregister_transformer("never_registered")

    @pytest.mark.parametrize("name", _BUILTINS)
    def test_builtin_cannot_be_removed(self, name, clean_registry):
        with pytest.raises(ValueError, match="Cannot unregister built-in"):
            unregister_transformer(name)
        assert name in _TRANSFORMER_REGISTRY

    def test_unregister_overridden_builtin_restores_default(self, clean_registry):
        default = _TRANSFORMER_REGISTRY["SNIPBaseline"]
        register_transformer("SNIPBaseline", Scale, override=True)
        assert _TRANSFORMER_REGISTRY["SNIPBaseline"] is Scale
        unregister_transformer("SNIPBaseline")
        assert _TRANSFORMER_REGISTRY["SNIPBaseline"] is default


class TestRoundTrip:
    def test_dict_round_trip_preserves_params(self, clean_registry, spectrum):
        register_transformer("Scale", Scale)
        pipe = PreprocessingPipeline([("scale", Scale(3.0))])
        rebuilt = PreprocessingPipeline.from_dict(pipe.to_dict())
        assert rebuilt.get_step("scale").factor == 3.0
        pd.testing.assert_frame_equal(rebuilt(spectrum), pipe(spectrum))

    def test_json_round_trip(self, clean_registry, tmp_path, spectrum):
        register_transformer("Scale", Scale)
        pipe = PreprocessingPipeline([("scale", Scale(4.0))])
        path = tmp_path / "pipeline.json"
        pipe.to_json(path)
        rebuilt = PreprocessingPipeline.from_json(path)
        assert rebuilt.get_step("scale").factor == 4.0
        pd.testing.assert_frame_equal(rebuilt(spectrum), pipe(spectrum))

    def test_yaml_round_trip(self, clean_registry, tmp_path, spectrum):
        register_transformer("Scale", Scale)
        pipe = PreprocessingPipeline([("scale", Scale(5.0))])
        path = tmp_path / "pipeline.yaml"
        pipe.to_yaml(path)
        rebuilt = PreprocessingPipeline.from_yaml(path)
        assert rebuilt.get_step("scale").factor == 5.0

    def test_mixed_builtin_and_custom_pipeline(self, clean_registry, spectrum):
        from maldiamrkit.preprocessing import ClipNegatives

        register_transformer("Scale", Scale)
        pipe = PreprocessingPipeline([("clip", ClipNegatives()), ("scale", Scale(2.0))])
        rebuilt = PreprocessingPipeline.from_dict(pipe.to_dict())
        assert rebuilt.step_names == ["clip", "scale"]
        pd.testing.assert_frame_equal(rebuilt(spectrum), pipe(spectrum))

    def test_config_is_json_serialisable(self, clean_registry):
        register_transformer("Scale", Scale)
        pipe = PreprocessingPipeline([("scale", Scale(3.0))])
        assert json.loads(json.dumps(pipe.to_dict())) == pipe.to_dict()


class TestRoundTripGuard:
    def test_warns_when_name_not_registered(self, clean_registry):
        pipe = PreprocessingPipeline([("scale", Scale(3.0))])
        with pytest.warns(UserWarning, match="not registered"):
            pipe.to_dict()

    def test_warns_when_declared_name_mismatches(self, clean_registry):
        register_transformer("WrongName", WrongName)
        pipe = PreprocessingPipeline([("bad", WrongName())])
        with pytest.warns(UserWarning, match="SomethingElse"):
            pipe.to_dict()

    def test_warns_when_name_key_missing(self, clean_registry):
        pipe = PreprocessingPipeline([("nameless", NoNameKey())])
        with pytest.warns(UserWarning, match="without a 'name' key"):
            pipe.to_dict()

    def test_no_warning_once_registered(self, clean_registry, recwarn):
        register_transformer("Scale", Scale)
        PreprocessingPipeline([("scale", Scale(3.0))]).to_dict()
        assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []

    def test_default_pipeline_never_warns(self, recwarn):
        PreprocessingPipeline.default().to_dict()
        assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []

    def test_dict_still_produced_despite_warning(self, clean_registry):
        pipe = PreprocessingPipeline([("scale", Scale(3.0))])
        with pytest.warns(UserWarning):
            d = pipe.to_dict()
        assert d["steps"][0]["factor"] == 3.0

    def test_warns_when_name_registered_to_different_class(self, clean_registry):
        class Impostor:
            def __call__(self, df):
                return df

            def to_dict(self):
                return {"name": "Scale"}

        register_transformer("Scale", Impostor)
        pipe = PreprocessingPipeline([("scale", Scale(3.0))])
        with pytest.warns(UserWarning, match="registered to Impostor"):
            pipe.to_dict()

    def test_warning_points_at_caller_via_to_json(self, clean_registry, tmp_path):
        pipe = PreprocessingPipeline([("scale", Scale(3.0))])
        with pytest.warns(UserWarning) as record:
            pipe.to_json(tmp_path / "p.json")
        assert "maldiamrkit" not in record[0].filename


class TestLoadErrors:
    def test_unknown_transformer_raises_value_error(self, clean_registry):
        with pytest.raises(ValueError, match="is not a registered transformer"):
            PreprocessingPipeline.from_dict(
                {"steps": [{"step_name": "s", "name": "Ghost"}]}
            )

    def test_error_points_at_register_transformer(self, clean_registry):
        with pytest.raises(ValueError, match="register_transformer"):
            PreprocessingPipeline.from_dict(
                {"steps": [{"step_name": "s", "name": "Ghost"}]}
            )

    def test_missing_steps_key_raises_value_error(self):
        with pytest.raises(ValueError, match="missing the 'steps' key"):
            PreprocessingPipeline.from_dict({})

    def test_missing_step_name_key_raises_value_error(self):
        with pytest.raises(ValueError, match="missing the 'step_name' key"):
            PreprocessingPipeline.from_dict({"steps": [{"name": "ClipNegatives"}]})

    def test_missing_name_key_raises_value_error(self):
        with pytest.raises(ValueError, match="missing the 'name' key"):
            PreprocessingPipeline.from_dict({"steps": [{"step_name": "s"}]})

    def test_unregistered_after_save_fails_to_load(self, clean_registry, tmp_path):
        register_transformer("Scale", Scale)
        path = tmp_path / "p.json"
        PreprocessingPipeline([("scale", Scale(3.0))]).to_json(path)
        unregister_transformer("Scale")
        with pytest.raises(ValueError, match="is not a registered transformer"):
            PreprocessingPipeline.from_json(path)
