"""
Regressionstest: Spannungsteiler R1=R2=10k, V1=5V DC.
Prüft Eingangs- und Ausgangsspannung aus der .op-Simulation.
"""

import pytest

from spice_analysis.models import SimulationResult


@pytest.mark.regression
@pytest.mark.spannungsteiler
class TestSpannungsteiler:
    def test_signale_vorhanden(self, spannungsteiler_raw: SimulationResult):
        """V(in) und V(out) müssen im Ergebnis enthalten sein."""
        namen = spannungsteiler_raw.signal_names()
        assert "V(in)" in namen,  f"V(in) fehlt  — vorhanden: {namen}"
        assert "V(out)" in namen, f"V(out) fehlt — vorhanden: {namen}"

    def test_eingangsspannung(self, spannungsteiler_raw: SimulationResult):
        """Eingangsspannung V(in) muss 5 V betragen."""
        v_in = float(spannungsteiler_raw.signals["V(in)"].iloc[0])
        assert v_in == pytest.approx(5.0, abs=0.01)

    def test_ausgangsspannung(self, spannungsteiler_raw: SimulationResult):
        """Ausgangsspannung V(out) muss bei R1=R2 genau halb sein: 2.5 V."""
        v_out = float(spannungsteiler_raw.signals["V(out)"].iloc[0])
        assert v_out == pytest.approx(2.5, abs=0.01)

    def test_teilungsverhaeltnis(self, spannungsteiler_raw: SimulationResult):
        """Teilungsverhältnis V(out)/V(in) muss 0.5 ergeben (R1=R2)."""
        v_in  = float(spannungsteiler_raw.signals["V(in)"].iloc[0])
        v_out = float(spannungsteiler_raw.signals["V(out)"].iloc[0])
        assert v_out / v_in == pytest.approx(0.5, abs=0.001)
