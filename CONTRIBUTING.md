# Contributing to MaldiAMRKit

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/EttoreRocchi/MaldiAMRKit.git
cd MaldiAMRKit
pip install -e .[dev]
```

## Running Tests

```bash
pytest
```

## Linting

```bash
ruff check .
```

## Submitting Changes

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Make sure all tests pass and the linter is clean.
4. Open a pull request with a clear title and description of your changes.

## Deprecation Policy

Before removing or renaming a public API, follow these steps:

1. **Deprecate first.** Add the `@deprecated` decorator from `maldiamrkit/_compat.py` so that callers receive a `DeprecationWarning` naming the replacement and the version where the old API will be removed.
2. **Keep the deprecated API working** for at least one minor release.
3. **Add a CHANGELOG entry** for both the deprecation and the eventual removal.
4. **Remove** the deprecated API in the planned version and record the removal in the CHANGELOG.

> **Pre-1.0:** one minor-release grace period (e.g. deprecate in 0.13, remove in 0.14). This policy will be revisited for longer cycles once the project reaches v1.0.

## Reporting Issues

Open an issue on [GitHub](https://github.com/EttoreRocchi/MaldiAMRKit/issues) with a minimal reproducible example if applicable.
