import csv
from pathlib import Path
import pandas as pd

def sniff_delimiter(path: str | Path, sample_lines: int = 10) -> str:
    with open(path, "r", newline="") as f:
        dialect = csv.Sniffer().sniff(
            "".join([next(f) for _ in range(sample_lines)]),
            delimiters=",;\t "
        )
    return dialect.delimiter

def read_spectrum(path: str | Path) -> pd.DataFrame:
    """
    Read raw txt/csv file with two unnamed columns into a DataFrame
    with columns ['mass', 'intensity'].
    """
    try:
        delim = sniff_delimiter(path)
    except:
        delim = "\s+"
    
    df = pd.read_csv(
        path,
        sep=delim,
        comment="#",
        header=None,
        names=["mass", "intensity"]
    )
    
    return df
