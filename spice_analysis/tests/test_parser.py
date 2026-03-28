"""
Regressionstests für den LTspice .raw Parser.
"""

from pathlib import Path
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
