"""File writing utilities for MALDI-TOF spectrum data."""

from __future__ import annotations

import base64
import gzip
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .readers import (
    _CV_FLOAT64,
    _CV_INTENSITY_ARRAY,
    _CV_MZ_ARRAY,
    _CV_ZLIB,
    _format_suffixes,
)

_MZML_NAMESPACE = "http://psi.hupo.org/ms/mzml"
_PSI_MS_OBO = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"


def _cv_param(parent: ET.Element, accession: str, name: str, **attrs: str) -> None:
    ET.SubElement(
        parent,
        "cvParam",
        {"cvRef": "MS", "accession": accession, "name": name, **attrs},
    )


def _binary_data_array(
    parent: ET.Element, values: np.ndarray, accession: str, name: str, **attrs: str
) -> None:
    raw = zlib.compress(np.asarray(values, dtype="<f8").tobytes())
    encoded = base64.b64encode(raw).decode("ascii")
    bda = ET.SubElement(parent, "binaryDataArray", {"encodedLength": str(len(encoded))})
    _cv_param(bda, accession, name, **attrs)
    _cv_param(bda, _CV_FLOAT64, "64-bit float")
    _cv_param(bda, _CV_ZLIB, "zlib compression")
    ET.SubElement(bda, "binary").text = encoded


def _write_mzml(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a spectrum to a single-spectrum mzML 1.1 file.

    Arrays are stored as zlib-compressed 64-bit floats, readable by
    :func:`~maldiamrkit.io.read_spectrum` with or without pyteomics.
    A trailing ``.gz`` suffix (e.g. ``.mzML.gz``) gzips the file.

    Parameters
    ----------
    df : pd.DataFrame
        Spectrum with columns ``'mass'`` and ``'intensity'``.
    path : str or Path
        Output file path.
    """
    from .. import __version__ as _maldiamrkit_version

    path = Path(path)
    mz = np.asarray(df["mass"], dtype=np.float64)
    intensity = np.asarray(df["intensity"], dtype=np.float64)

    root = ET.Element("mzML", {"xmlns": _MZML_NAMESPACE, "version": "1.1.0"})
    cv_list = ET.SubElement(root, "cvList", {"count": "1"})
    ET.SubElement(
        cv_list,
        "cv",
        {
            "id": "MS",
            "fullName": "Proteomics Standards Initiative Mass Spectrometry Ontology",
            "URI": _PSI_MS_OBO,
        },
    )
    file_content = ET.SubElement(ET.SubElement(root, "fileDescription"), "fileContent")
    _cv_param(file_content, "MS:1000579", "MS1 spectrum")

    software = ET.SubElement(
        ET.SubElement(root, "softwareList", {"count": "1"}),
        "software",
        {"id": "maldiamrkit", "version": _maldiamrkit_version},
    )
    _cv_param(
        software, "MS:1000799", "custom unreleased software tool", value="MaldiAMRKit"
    )

    instrument = ET.SubElement(
        ET.SubElement(root, "instrumentConfigurationList", {"count": "1"}),
        "instrumentConfiguration",
        {"id": "IC1"},
    )
    _cv_param(instrument, "MS:1000031", "instrument model")

    processing = ET.SubElement(
        ET.SubElement(
            ET.SubElement(root, "dataProcessingList", {"count": "1"}),
            "dataProcessing",
            {"id": "maldiamrkit_export"},
        ),
        "processingMethod",
        {"order": "0", "softwareRef": "maldiamrkit"},
    )
    _cv_param(processing, "MS:1000544", "Conversion to mzML")

    run = ET.SubElement(
        root, "run", {"id": "run", "defaultInstrumentConfigurationRef": "IC1"}
    )
    spectrum_list = ET.SubElement(
        run,
        "spectrumList",
        {"count": "1", "defaultDataProcessingRef": "maldiamrkit_export"},
    )
    spectrum = ET.SubElement(
        spectrum_list,
        "spectrum",
        {"index": "0", "id": "scan=1", "defaultArrayLength": str(len(mz))},
    )
    _cv_param(spectrum, "MS:1000579", "MS1 spectrum")
    _cv_param(spectrum, "MS:1000511", "ms level", value="1")
    _cv_param(spectrum, "MS:1000128", "profile spectrum")
    arrays = ET.SubElement(spectrum, "binaryDataArrayList", {"count": "2"})
    _binary_data_array(
        arrays,
        mz,
        _CV_MZ_ARRAY,
        "m/z array",
        unitCvRef="MS",
        unitAccession="MS:1000040",
        unitName="m/z",
    )
    _binary_data_array(arrays, intensity, _CV_INTENSITY_ARRAY, "intensity array")

    ET.indent(root)
    _, gzipped = _format_suffixes(path)
    with gzip.open(path, "wb") if gzipped else open(path, "wb") as fh:
        ET.ElementTree(root).write(fh, encoding="utf-8", xml_declaration=True)
