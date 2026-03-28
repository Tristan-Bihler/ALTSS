"""
Elektrische Analysefunktionen für SimulationResult-Daten.
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

from .models import SimulationResult


def berechne_rms(signal: pd.Series) -> float:
    """Effektivwert (RMS) eines Signals.

    Args:
        signal: Zeitreihendaten als pandas Series.

    Returns:
        RMS-Wert in der Einheit des Signals.

    Raises:
        ValueError: Signal ist leer.
    """
    if signal.empty:
        raise ValueError("Signal darf nicht leer sein.")
    return float(np.sqrt(np.mean(signal.to_numpy() ** 2)))


def berechne_thd(
    result: SimulationResult,
    signal_name: str,
    grundfrequenz_hz: float,
    n_harmonische: int = 5,
) -> float:
    """Total Harmonic Distortion (THD) in Prozent.

    Args:
        result: Simulationsergebnis.
        signal_name: Name des zu analysierenden Signals.
        grundfrequenz_hz: Grundfrequenz in Hz.
        n_harmonische: Anzahl Oberwellen (ohne Grundwelle).

    Returns:
        THD in Prozent.

    Raises:
        KeyError: Signalname nicht in Ergebnissen.
        ValueError: Ungültige Frequenz oder Anzahl Oberwellen.
    """
    if signal_name not in result.signals.columns:
        raise KeyError(f"Signal '{signal_name}' nicht gefunden.")
    if grundfrequenz_hz <= 0:
        raise ValueError("Grundfrequenz muss positiv sein.")
    if n_harmonische < 1:
        raise ValueError("Mindestens eine Oberwelle erforderlich.")

    signal = result.signals[signal_name].to_numpy()
    zeitachse = result.signals.index.to_numpy(dtype=float)
    dt = float(np.mean(np.diff(zeitachse)))

    spektrum = np.abs(rfft(signal))
    freqachse = rfftfreq(len(signal), d=dt)

    def amplitude_bei(freq_hz: float) -> float:
        idx = int(np.argmin(np.abs(freqachse - freq_hz)))
        return float(spektrum[idx])

    a1 = amplitude_bei(grundfrequenz_hz)
    if a1 == 0.0:
        return 0.0

    oberwellen_energie = sum(
        amplitude_bei(grundfrequenz_hz * k) ** 2
        for k in range(2, n_harmonische + 2)
    )
    return float(np.sqrt(oberwellen_energie) / a1 * 100.0)


def fft_spektrum(
    result: SimulationResult,
    signal_name: str,
) -> pd.DataFrame:
    """FFT-Spektrum eines Signals.

    Args:
        result: Simulationsergebnis.
        signal_name: Signalname.

    Returns:
        DataFrame mit Spalten 'frequenz_hz' und 'amplitude'.
    """
    if signal_name not in result.signals.columns:
        raise KeyError(f"Signal '{signal_name}' nicht gefunden.")

    signal = result.signals[signal_name].to_numpy()
    dt = float(np.mean(np.diff(result.signals.index.to_numpy(dtype=float))))

    amplitude = np.abs(rfft(signal))
    frequenz = rfftfreq(len(signal), d=dt)

    return pd.DataFrame({"frequenz_hz": frequenz, "amplitude": amplitude})
