"""
LTspice .raw-Datei Parser (binär & ASCII).
Unterstützt Transient- und AC-Simulationen.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .models import SimulationResult


def parse_raw(path: str | Path) -> SimulationResult:
    """Liest eine LTspice .raw-Datei und gibt ein SimulationResult zurück.

    Args:
        path: Pfad zur .raw-Datei.

    Returns:
        SimulationResult mit Signalen als DataFrame.

    Raises:
        FileNotFoundError: Datei existiert nicht.
        ValueError: Unbekanntes Dateiformat.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    with path.open("rb") as fh:
        raw_bytes = fh.read()

    # Trenne Header (ASCII) vom Datenteil
    header_end = raw_bytes.find(b"Binary:\n")
    if header_end == -1:
        header_end = raw_bytes.find(b"Values:\n")
        binary_mode = False
    else:
        binary_mode = True

    if header_end == -1:
        raise ValueError(f"Unbekanntes .raw-Format: {path}")

    header_text = raw_bytes[:header_end].decode("latin-1")
    metadata, variable_names = _parse_header(header_text)

    offset = header_end + len("Binary:\n" if binary_mode else "Values:\n")
    data_bytes = raw_bytes[offset:]

    if binary_mode:
        signals_df = _parse_binary(data_bytes, variable_names, metadata)
    else:
        signals_df = _parse_ascii(data_bytes, variable_names)

    return SimulationResult(
        source_file=str(path),
        signals=signals_df,
        metadata=metadata,
    )


def _parse_header(text: str) -> tuple[dict, list[str]]:
    metadata: dict = {}
    variable_names: list[str] = []
    in_vars = False

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("variables:"):
            in_vars = True
            continue
        if in_vars:
            parts = line.split()
            if len(parts) >= 3:
                variable_names.append(parts[1])
        elif ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip().lower()] = value.strip()

    return metadata, variable_names


def _parse_binary(
    data: bytes, names: list[str], metadata: dict
) -> pd.DataFrame:
    n_vars = int(metadata.get("no. variables", len(names)))
    dtype = np.float64
    values = np.frombuffer(data, dtype=dtype)
    n_points = len(values) // n_vars
    if n_points == 0:
        return pd.DataFrame(columns=names)
    matrix = values[: n_points * n_vars].reshape(n_points, n_vars)
    df = pd.DataFrame(matrix, columns=names)
    if names:
        df = df.set_index(names[0])
        df.index.name = "time_s"
    return df


def _parse_ascii(data: bytes, names: list[str]) -> pd.DataFrame:
    rows: list[list[float]] = []
    for line in data.decode("latin-1").splitlines():
        parts = line.split()
        if len(parts) == len(names):
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=names) if rows else pd.DataFrame(columns=names)
    if names:
        df = df.set_index(names[0])
        df.index.name = "time_s"
    return df
