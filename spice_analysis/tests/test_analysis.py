"""
Regressionstests für Analysefunktionen.
"""

import numpy as np
import pandas as pd
import pytest

from spice_analysis.analysis import berechne_rms, berechne_thd, fft_spektrum
from spice_analysis.models import SimulationResult


@pytest.mark.unit
class TestRMS:
    def test_sinus_rms(self, sinus_result: SimulationResult):
        """RMS eines 1V-Sinus ≈ 1/√2."""
        rms = berechne_rms(sinus_result.signals["V(out)"])
        assert rms == pytest.approx(1 / np.sqrt(2), rel=1e-2)

    def test_leeres_signal_wirft_fehler(self, sinus_result: SimulationResult):
        with pytest.raises(ValueError):
            berechne_rms(pd.Series([], dtype=float))


@pytest.mark.unit
class TestTHD:
    def test_reiner_sinus_thd_niedrig(self, sinus_result: SimulationResult):
        """Reiner Sinus hat THD < 1 %."""
        thd = berechne_thd(sinus_result, "V(out)", grundfrequenz_hz=1_000.0)
        assert thd < 1.0

    def test_unbekanntes_signal_wirft_fehler(self, sinus_result: SimulationResult):
        with pytest.raises(KeyError):
            berechne_thd(sinus_result, "V(nichtvorhanden)", grundfrequenz_hz=1_000.0)

    def test_nullsignal_thd_ist_null(self, sinus_result: SimulationResult):
        """Wenn Grundwelle = 0, soll THD 0.0 zurückgeben (kein ZeroDivision)."""
        t = np.linspace(0, 1e-3, 500)
        df = pd.DataFrame({"V(dc)": np.zeros(500)}, index=pd.Index(t, name="time_s"))
        result = SimulationResult(source_file="dc", signals=df)
        assert berechne_thd(result, "V(dc)", grundfrequenz_hz=1_000.0) == 0.0

    def test_negative_frequenz_wirft_fehler(self, sinus_result: SimulationResult):
        with pytest.raises(ValueError):
            berechne_thd(sinus_result, "V(out)", grundfrequenz_hz=-1.0)


@pytest.mark.unit
class TestFFT:
    def test_grundfrequenz_dominant(self, sinus_result: SimulationResult):
        """Bei 1 kHz Sinus muss 1 kHz die stärkste Komponente sein."""
        spektrum = fft_spektrum(sinus_result, "V(out)")
        peak_freq = spektrum.loc[spektrum["amplitude"].idxmax(), "frequenz_hz"]
        assert peak_freq == pytest.approx(1_000.0, rel=0.05)

    def test_ausgabe_hat_korrekte_spalten(self, sinus_result: SimulationResult):
        df = fft_spektrum(sinus_result, "V(out)")
        assert set(df.columns) == {"frequenz_hz", "amplitude"}

    def test_unbekanntes_signal_wirft_fehler(self, sinus_result: SimulationResult):
        with pytest.raises(KeyError):
            fft_spektrum(sinus_result, "V(nichtvorhanden)")
