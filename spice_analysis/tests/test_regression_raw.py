"""
Regressionstests gegen echte LTspice .raw-Dateien.
Werden übersprungen wenn .raw-Dateien noch nicht vorhanden sind.
"""

import numpy as np
import pytest

from spice_analysis.analysis import berechne_rms, berechne_thd, fft_spektrum
from spice_analysis.models import SimulationResult


@pytest.mark.regression
class TestRCTiefpass1Ord:
    def test_signale_vorhanden(self, rc_tiefpass_raw: SimulationResult):
        assert "V(out)" in rc_tiefpass_raw.signal_names()
        assert len(rc_tiefpass_raw.signals) > 100

    def test_ausgangsamplitude_bei_1khz(self, rc_tiefpass_raw: SimulationResult):
        """Bei f = f_g = 1 kHz: Ausgangspegel ≈ -3 dB."""
        rms_out = berechne_rms(rc_tiefpass_raw.signals["V(out)"])
        rms_ein = 1.0 / np.sqrt(2)
        pegel_db = 20 * np.log10(rms_out / rms_ein)
        assert pegel_db == pytest.approx(-3.0, abs=0.5)

    def test_grundfrequenz_dominant(self, rc_tiefpass_raw: SimulationResult):
        spektrum = fft_spektrum(rc_tiefpass_raw, "V(out)")
        peak_freq = spektrum.loc[spektrum["amplitude"].idxmax(), "frequenz_hz"]
        assert peak_freq == pytest.approx(1_000.0, rel=0.05)

    def test_thd_niedrig(self, rc_tiefpass_raw: SimulationResult):
        thd = berechne_thd(rc_tiefpass_raw, "V(out)", grundfrequenz_hz=1_000.0)
        assert thd < 1.0


@pytest.mark.regression
class TestRCTiefpass2Ord:
    def test_signale_vorhanden(self, rc_tiefpass_2ord_raw: SimulationResult):
        assert "V(out)" in rc_tiefpass_2ord_raw.signal_names()

    def test_staerkere_daempfung_als_1ord(
        self,
        rc_tiefpass_raw: SimulationResult,
        rc_tiefpass_2ord_raw: SimulationResult,
    ):
        """2. Ordnung muss stärker dämpfen als 1. Ordnung."""
        rms_1ord = berechne_rms(rc_tiefpass_raw.signals["V(out)"])
        rms_2ord = berechne_rms(rc_tiefpass_2ord_raw.signals["V(out)"])
        assert rms_2ord < rms_1ord


@pytest.mark.regression
class TestLCResonanz:
    def test_signale_vorhanden(self, lc_resonanz_raw: SimulationResult):
        assert "V(out)" in lc_resonanz_raw.signal_names()
        assert len(lc_resonanz_raw.signals) > 100

    def test_resonanzfrequenz(self, lc_resonanz_raw: SimulationResult):
        """f_r = 1/(2π√(LC)) ≈ 1 kHz."""
        spektrum = fft_spektrum(lc_resonanz_raw, "V(out)")
        peak_freq = spektrum.loc[spektrum["amplitude"].idxmax(), "frequenz_hz"]
        assert peak_freq == pytest.approx(1_000.0, rel=0.1)

    def test_amplitude_bei_resonanz_hoch(self, lc_resonanz_raw: SimulationResult):
        rms = berechne_rms(lc_resonanz_raw.signals["V(out)"])
        assert rms > 1.0 / np.sqrt(2)
