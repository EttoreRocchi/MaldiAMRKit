"""Shared pytest fixtures for MaldiAMRKit tests."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Path to the data directory
DATA_DIR = Path(__file__).parent.parent / "data"
SPECTRA_DIR = DATA_DIR  # Spectra are directly in data/
METADATA_FILE = DATA_DIR / "metadata" / "metadata.csv"


def _generate_synthetic_spectrum(
    mz_start: float = 2000,
    mz_end: float = 20000,
    n_points: int = 18000,
    peak_positions: list[float] | None = None,
    peak_heights: list[float] | None = None,
    peak_width: float = 10.0,
    noise_level: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic MALDI-TOF spectrum with known peaks.

    Parameters
    ----------
    mz_start : float
        Start of m/z range.
    mz_end : float
        End of m/z range.
    n_points : int
        Number of data points.
    peak_positions : list of float, optional
        m/z positions of peaks. Defaults to common positions.
    peak_heights : list of float, optional
        Heights of peaks. Defaults to 1000 for all peaks.
    peak_width : float
        Standard deviation of Gaussian peaks.
    noise_level : float
        Standard deviation of Gaussian noise.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'mass' and 'intensity' columns.
    """
    rng = np.random.default_rng(random_state)

    mz = np.linspace(mz_start, mz_end, n_points)
    intensity = np.zeros_like(mz)

    if peak_positions is None:
        peak_positions = [3000, 5000, 7500, 10000, 12500, 15000]
    if peak_heights is None:
        peak_heights = [1000.0] * len(peak_positions)

    # Add Gaussian peaks
    for pos, height in zip(peak_positions, peak_heights):
        idx = np.argmin(np.abs(mz - pos))
        intensity += height * np.exp(-0.5 * ((mz - mz[idx]) / peak_width) ** 2)

    # Add noise
    intensity += rng.normal(0, noise_level, len(intensity))
    intensity = np.maximum(intensity, 0)  # Clip negatives

    return pd.DataFrame({"mass": mz, "intensity": intensity})


@pytest.fixture
def synthetic_spectrum() -> pd.DataFrame:
    """
    Generate a synthetic MALDI-TOF spectrum with known peaks.

    Peaks are at m/z: 3000, 5000, 7500, 10000, 12500, 15000
    """
    return _generate_synthetic_spectrum(random_state=42)


@pytest.fixture
def synthetic_spectrum_shifted() -> pd.DataFrame:
    """
    Generate a synthetic spectrum with systematic m/z shift.

    Same peaks as synthetic_spectrum but shifted by +5 Da.
    """
    df = _generate_synthetic_spectrum(random_state=42)
    df["mass"] = df["mass"] + 5.0
    return df


@pytest.fixture
def synthetic_spectrum_noisy() -> pd.DataFrame:
    """Generate a synthetic spectrum with higher noise level."""
    return _generate_synthetic_spectrum(noise_level=100.0, random_state=42)


@pytest.fixture
def binned_dataset() -> pd.DataFrame:
    """
    Generate a synthetic binned dataset for transformer testing.

    Returns a DataFrame with 50 samples and ~6000 features (bins).
    Uses random_state for reproducibility.
    """
    rng = np.random.default_rng(42)

    n_samples = 50
    mz_start = 2000
    mz_end = 20000
    bin_width = 3
    n_bins = int((mz_end - mz_start) / bin_width) + 1

    # Generate base intensities (exponential distribution)
    X = rng.exponential(0.001, (n_samples, n_bins))

    # Add peak patterns at random positions for each sample
    for i in range(n_samples):
        peak_bins = rng.choice(n_bins, size=20, replace=False)
        X[i, peak_bins] += rng.uniform(0.01, 0.1, 20)

    # Normalize each sample to sum to 1
    X = X / X.sum(axis=1, keepdims=True)

    # Create column names as m/z values
    columns = [str(mz_start + i * bin_width) for i in range(n_bins)]
    index = [f"sample_{i}" for i in range(n_samples)]

    return pd.DataFrame(X, columns=columns, index=index)


@pytest.fixture
def binned_dataset_with_shift(binned_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Binned dataset with systematic shift in every other sample.

    Useful for testing alignment methods.
    """
    X = binned_dataset.copy()
    # Shift every other sample by 2 bins
    X.iloc[::2] = np.roll(X.iloc[::2].values, 2, axis=1)
    return X


@pytest.fixture
def real_spectrum_path() -> Path:
    """Path to a real spectrum file from the data directory."""
    spectrum_files = sorted(SPECTRA_DIR.glob("*.txt"))
    if not spectrum_files:
        pytest.skip("No spectrum files found in data directory")
    return spectrum_files[0]


@pytest.fixture
def real_spectra_paths() -> list[Path]:
    """List of paths to real spectrum files (first 5)."""
    spectrum_files = sorted(SPECTRA_DIR.glob("*.txt"))[:5]
    if not spectrum_files:
        pytest.skip("No spectrum files found in data directory")
    return spectrum_files


@pytest.fixture
def data_dir() -> Path:
    """Path to the data directory."""
    if not DATA_DIR.exists():
        pytest.skip("Data directory not found")
    return DATA_DIR


@pytest.fixture
def spectra_dir() -> Path:
    """Path to the spectra directory."""
    if not SPECTRA_DIR.exists():
        pytest.skip("Spectra directory not found")
    return SPECTRA_DIR


@pytest.fixture
def metadata_file() -> Path:
    """Path to the metadata CSV file."""
    if not METADATA_FILE.exists():
        pytest.skip("Metadata file not found")
    return METADATA_FILE


@pytest.fixture
def classification_labels(binned_dataset: pd.DataFrame) -> np.ndarray:
    """Binary classification labels for the binned dataset."""
    rng = np.random.default_rng(42)
    return rng.choice([0, 1], size=len(binned_dataset))
