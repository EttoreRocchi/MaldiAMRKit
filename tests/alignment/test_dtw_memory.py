"""Memory of DTW alignment grows with the band, not with the square of n."""

import subprocess
import sys

import pytest

_CODE = """
import resource
import numpy as np
from maldiamrkit.alignment.strategies import DTWStrategy

rng = np.random.default_rng(0)
reference = rng.random(20000)
query = np.roll(reference, 3)
strategy = DTWStrategy(10)
strategy._dtw_align(query[:100], reference[:100])
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
strategy._dtw_align(query, reference)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(after - before)
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="ru_maxrss is in KiB on Linux"
)
def test_dtw_on_a_long_series_stays_small():
    out = subprocess.run(
        [sys.executable, "-c", _CODE], capture_output=True, text=True, check=True
    )
    increase_kib = int(out.stdout.strip().splitlines()[-1])
    assert increase_kib < 100 * 1024
