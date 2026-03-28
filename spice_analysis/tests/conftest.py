"""
Gemeinsame pytest Fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from spice_analysis.models import SimulationResult


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
