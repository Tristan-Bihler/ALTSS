"""
Gemeinsame pytest Fixtures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spice_analysis.models import SimulationResult
from spice_analysis.parser import parse_raw

FIXTURE_DIR = Path(__file__).parent.parent / "data" / "fixtures"


@pytest.fixture
def sinus_result() -> SimulationResult:
    """Sinus 1 kHz, 1 V_peak, 1 ms Zeitraum, 10 kHz Abtastrate."""
    fs = 10_000.0
    t = np.arange(0, 1e-3, 1 / fs)
    v = np.sin(2 * np.pi * 1_000 * t)
    df = pd.DataFrame({"V(out)": v}, index=pd.Index(t, name="time_s"))
    return SimulationResult(source_file="fixture", signals=df)


@pytest.fixture
def rc_result() -> SimulationResult:
    """Simuliertes RC-Tiefpass-Signal (exponentieller Abfall)."""
    t = np.linspace(0, 1e-3, 500)
    tau = 1e-4
    v = np.exp(-t / tau)
    df = pd.DataFrame({"V(out)": v}, index=pd.Index(t, name="time_s"))
    return SimulationResult(source_file="rc_fixture", signals=df)


@pytest.fixture(scope="session")
def rc_tiefpass_raw() -> SimulationResult:
    """Echte LTspice .raw-Datei: RC-Tiefpass 1. Ordnung (1 kHz, 1 Vpeak)."""
    path = FIXTURE_DIR / "rc_tiefpass.raw"
    if not path.exists():
        pytest.skip(f"Fixture nicht vorhanden: {path} — erst LTspice-Simulation ausführen.")
    return parse_raw(path)


@pytest.fixture(scope="session")
def rc_tiefpass_2ord_raw() -> SimulationResult:
    """Echte LTspice .raw-Datei: RC-Tiefpass 2. Ordnung."""
    path = FIXTURE_DIR / "rc_tiefpass_2ord.raw"
    if not path.exists():
        pytest.skip(f"Fixture nicht vorhanden: {path} — erst LTspice-Simulation ausführen.")
    return parse_raw(path)


@pytest.fixture(scope="session")
def lc_resonanz_raw() -> SimulationResult:
    """Echte LTspice .raw-Datei: LC-Parallelschwingkreis."""
    path = FIXTURE_DIR / "lc_resonanz.raw"
    if not path.exists():
        pytest.skip(f"Fixture nicht vorhanden: {path} — erst LTspice-Simulation ausführen.")
    return parse_raw(path)


@pytest.fixture(scope="session")
def spannungsteiler_raw() -> SimulationResult:
    """Echte LTspice .raw-Datei: Spannungsteiler R1=R2=10k, V1=5V DC."""
    path = FIXTURE_DIR / "spannungsteiler.raw"
    if not path.exists():
        pytest.skip(f"Fixture nicht vorhanden: {path} — erst LTspice-Simulation ausführen.")
    return parse_raw(path)
