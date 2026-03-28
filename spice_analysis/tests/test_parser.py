"""
Regressionstests für den LTspice .raw Parser.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from spice_analysis.parser import parse_raw


def test_parse_fehlende_datei():
    with pytest.raises(FileNotFoundError):
        parse_raw("nicht_vorhanden.raw")


def test_parse_ungueltige_datei(tmp_path: Path):
    f = tmp_path / "leer.raw"
    f.write_bytes(b"kein gueltiger header")
    with pytest.raises(ValueError, match="Unbekanntes .raw-Format"):
        parse_raw(f)


def test_parse_ascii_raw(tmp_path: Path):
    """Minimale ASCII .raw-Datei mit zwei Datenpunkten."""
    inhalt = (
        "Title: Test\n"
        "No. Variables: 2\n"
        "Variables:\n"
        "\t0\ttime\ttime\n"
        "\t1\tV(out)\tvoltage\n"
        "Values:\n"
        "0.0 1.0\n"
        "1e-6 0.9\n"
    )
    f = tmp_path / "test.raw"
    f.write_bytes(inhalt.encode("latin-1"))
    result = parse_raw(f)

    assert "V(out)" in result.signal_names()
    assert len(result.signals) == 2
    assert result.signals["V(out)"].iloc[0] == pytest.approx(1.0)


def _erstelle_binary_raw(tmp_path: Path) -> Path:
    """Minimale gültige Binär-.raw-Datei mit 3 Zeitpunkten."""
    header = (
        "Title: BinärTest\n"
        "No. Variables: 2\n"
        "Variables:\n"
        "\t0\ttime\ttime\n"
        "\t1\tV(out)\tvoltage\n"
        "Binary:\n"
    ).encode("latin-1")
    daten = struct.pack("6d", 0.0, 1.0, 1e-6, 0.9, 2e-6, 0.8)
    f = tmp_path / "binaer.raw"
    f.write_bytes(header + daten)
    return f


def test_parse_binaer_raw(tmp_path: Path):
    """Binärformat wird korrekt geparst."""
    f = _erstelle_binary_raw(tmp_path)
    result = parse_raw(f)

    assert "V(out)" in result.signal_names()
    assert len(result.signals) == 3
    assert result.signals["V(out)"].iloc[0] == pytest.approx(1.0)
    assert result.signals["V(out)"].iloc[2] == pytest.approx(0.8)


def test_parse_ascii_ohne_datenpunkte(tmp_path: Path):
    """ASCII-Datei mit Header aber ohne Messwerte → leerer DataFrame."""
    inhalt = (
        "Title: Leer\n"
        "No. Variables: 2\n"
        "Variables:\n"
        "\t0\ttime\ttime\n"
        "\t1\tV(out)\tvoltage\n"
        "Values:\n"
    )
    f = tmp_path / "leer_werte.raw"
    f.write_bytes(inhalt.encode("latin-1"))
    result = parse_raw(f)

    assert result.signals.empty
    assert "V(out)" in result.signal_names()
