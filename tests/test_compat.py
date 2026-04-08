"""Tests for the @deprecated decorator in maldiamrkit._compat."""

from __future__ import annotations

import warnings

import pytest

from maldiamrkit._compat import deprecated


@deprecated(new_name="new_add", removed_in="99.0")
def _old_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestDeprecatedDecorator:
    """Suite for :func:`maldiamrkit._compat.deprecated`."""

    def test_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _old_add(1, 2)

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) == 1
        msg = str(dep_warnings[0].message)
        assert "_old_add" in msg
        assert "new_add" in msg
        assert "v99.0" in msg

    def test_returns_correct_value(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert _old_add(3, 4) == 7

    def test_preserves_name(self):
        assert _old_add.__name__ == "_old_add"

    def test_preserves_docstring(self):
        assert _old_add.__doc__ == "Add two numbers."

    def test_warning_category_is_deprecation(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            _old_add(0, 0)
