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

    # UTF-16 LE erkennen (LTspice .op und neuere Versionen)
    if raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff") or raw_bytes[1:2] == b"\x00":
        encoding = "utf-16-le" if raw_bytes[1:2] == b"\x00" else "utf-16"
        text = raw_bytes.decode(encoding, errors="replace")
        return _parse_utf16_binary(text, raw_bytes, encoding, str(path))

    # UTF-8 / latin-1 Pfad (ältere .tran Dateien)
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


def _parse_utf16_binary(
    text: str, raw_bytes: bytes, encoding: str, source: str
) -> SimulationResult:
    """Parst UTF-16 .raw-Dateien mit Binärdatenteil (LTspice .op/.tran)."""
    lines = text.splitlines()
    metadata: dict = {}
    variable_names: list[str] = []
    mode = "header"
    header_char_len = 0

    for line in lines:
        line_s = line.strip()
        if line_s.lower().startswith("variables:"):
            mode = "variables"
            header_char_len += len(line) + 1
            continue
        if line_s.lower().startswith("binary:"):
            header_char_len += len(line) + 1
            break
        header_char_len += len(line) + 1

        if mode == "header" and ":" in line_s:
            key, _, value = line_s.partition(":")
            metadata[key.strip().lower()] = value.strip()
        elif mode == "variables":
            parts = line_s.split()
            if len(parts) >= 2:
                variable_names.append(parts[1])

    n_vars = len(variable_names)
    n_points = int(metadata.get("no. points", "1").strip() or "1")

    # Binärteil lokalisieren: Binary:\n in UTF-16
    marker = "Binary:\n".encode(encoding)
    bin_offset = raw_bytes.find(marker)
    if bin_offset == -1:
        raise ValueError(f"Kein Binary-Block gefunden: {source}")
    bin_offset += len(marker)

    data_bytes = raw_bytes[bin_offset:]
    expected = n_vars * n_points

    # LTspice .op Layout: erster Wert float64, alle weiteren float32
    # Beispiel: V(in)=float64(8b), V(out)=float32(4b), I(x)=float32(4b)
    values_64 = np.frombuffer(data_bytes, dtype=np.float64)
    if len(values_64) >= expected:
        # Alles float64 (älteres Format)
        matrix = values_64[:expected].reshape(n_points, n_vars)
        df = pd.DataFrame(matrix, columns=variable_names)
    else:
        # Gemischtes Layout: erster Wert float64, Rest float32
        rows = []
        for _ in range(n_points):
            row: list[float] = []
            offset = 0
            for i, _name in enumerate(variable_names):
                if i == 0:
                    size, dtype = 8, np.float64
                else:
                    size, dtype = 4, np.float32
                if offset + size <= len(data_bytes):
                    val = float(np.frombuffer(data_bytes[offset:offset + size], dtype=dtype)[0])
                    row.append(val)
                    offset += size
                else:
                    row.append(0.0)
            rows.append(row)
        df = pd.DataFrame(rows, columns=variable_names) if rows else pd.DataFrame(columns=variable_names)

    return SimulationResult(source_file=source, signals=df, metadata=metadata)


def _parse_utf16(text: str, source: str) -> SimulationResult:
    """Parst UTF-16 kodierte .raw-Dateien (LTspice .op und neuere Versionen).

    Das Format enthält Signalwerte als Schlüssel-Wert-Paare nach 'Values:'.
    """
    lines = text.splitlines()

    metadata: dict = {}
    variable_names: list[str] = []
    values: dict[str, float] = {}

    mode = "header"
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue

        if line_s.lower().startswith("variables:"):
            mode = "variables"
            continue
        if line_s.lower().startswith("values:"):
            mode = "values"
            continue
        if line_s.lower().startswith("binary:"):
            break

        if mode == "header" and ":" in line_s:
            key, _, value = line_s.partition(":")
            metadata[key.strip().lower()] = value.strip()

        elif mode == "variables":
            parts = line_s.split()
            if len(parts) >= 2:
                variable_names.append(parts[1])

        elif mode == "values":
            # Format: "idx\tV(name)\tWert" oder nur Wert auf eigener Zeile
            parts = line_s.split()
            if len(parts) >= 3:
                # z.B. "0\tV(in)\t5"
                try:
                    name = parts[1]
                    val = float(parts[2])
                    values[name] = val
                except (ValueError, IndexError):
                    pass
            elif len(parts) == 1:
                # Nur Zahlenwert — Index aus variable_names ableiten
                idx = len(values)
                if idx < len(variable_names):
                    try:
                        values[variable_names[idx]] = float(parts[0])
                    except ValueError:
                        pass

    if not values and variable_names:
        raise ValueError(f"Unbekanntes .raw-Format: {source}")

    df = pd.DataFrame([values])
    return SimulationResult(
        source_file=source,
        signals=df,
        metadata=metadata,
    )
