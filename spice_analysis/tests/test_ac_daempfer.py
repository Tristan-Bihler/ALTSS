"""
Regressionstest: AC-Daempfer 10Vp Sinus -> ~5Vp Sinus bei 1kHz.
Topologie: Resistiver Spannungsteiler + NPN Emitterfolger (BC547-aehnlich).
"""

import numpy as np
import pytest

from spice_analysis.analysis import berechne_rms, fft_spektrum
from spice_analysis.models import SimulationResult


def _eingeschwungen(result: SimulationResult, signal: str) -> "pd.Series":
    """Gibt nur die letzten 3 ms zurueck (eingeschwungener Zustand)."""
    import pandas as pd
    s = result.signals[signal]
    # Zeitachse aus Index oder separater Spalte
    if "time" in result.signals.columns:
        t = result.signals["time"]
    else:
        t = result.signals.index.to_series()
    maske = t >= t.max() - 5e-3
    return s[maske]


@pytest.mark.regression
@pytest.mark.ac_daempfer
class TestAcDaempfer:
    def test_signale_vorhanden(self, ac_daempfer_raw: SimulationResult):
        """V(out) und V(vin) muessen im Ergebnis enthalten sein."""
        namen = ac_daempfer_raw.signal_names()
        assert "V(out)" in namen, f"V(out) fehlt  — vorhanden: {namen}"
        assert "V(vin)" in namen, f"V(vin) fehlt  — vorhanden: {namen}"

    def test_eingangsamplitude_unveraendert(self, ac_daempfer_raw: SimulationResult):
        """Eingangs-RMS muss ~7.07 V betragen (10Vp, eingeschwungen)."""
        rms_in = berechne_rms(_eingeschwunden(ac_daempfer_raw, "V(vin)"))
        assert rms_in == pytest.approx(10.0 / np.sqrt(2), rel=0.10), (
            f"Eingangs-RMS = {rms_in:.3f} V, erwartet ~7.07 V"
        )

    def test_ausgangsamplitude(self, ac_daempfer_raw: SimulationResult):
        """Ausgangs-RMS muss ~3.54 V betragen (5Vp / sqrt(2), eingeschwungen)."""
        rms_out = berechne_rms(_eingeschwunden(ac_daempfer_raw, "V(out)"))
        assert rms_out == pytest.approx(5.0 / np.sqrt(2), rel=0.15), (
            f"Ausgangs-RMS = {rms_out:.3f} V, erwartet ~3.54 V"
        )

    def test_daempfungsverhaeltnis(self, ac_daempfer_raw: SimulationResult):
        """Daempfung V(out)/V(vin) muss zwischen 0.35 und 0.65 liegen."""
        rms_in  = berechne_rms(_eingeschwunden(ac_daempfer_raw, "V(vin)"))
        rms_out = berechne_rms(_eingeschwunden(ac_daempfer_raw, "V(out)"))
        verhaeltnis = rms_out / rms_in
        assert 0.35 <= verhaeltnis <= 0.65, (
            f"Daempfung = {verhaeltnis:.3f}, erwartet 0.35..0.65"
        )

    def test_grundfrequenz_erhalten(self, ac_daempfer_raw: SimulationResult):
        """FFT-Peak des Ausgangs muss bei 1 kHz liegen."""
        spektrum = fft_spektrum(ac_daempfer_raw, "V(out)")
        # DC-Anteil ignorieren (Index 0)
        spektrum_ohne_dc = spektrum.iloc[1:]
        peak_freq = spektrum_ohne_dc.loc[
            spektrum_ohne_dc["amplitude"].idxmax(), "frequenz_hz"
        ]
        assert peak_freq == pytest.approx(1_000.0, rel=0.10), (
            f"Peak-Frequenz = {peak_freq:.1f} Hz, erwartet 1000 Hz"
        )
