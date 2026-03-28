from dataclasses import dataclass, field
import pandas as pd


@dataclass
class SimulationResult:
    """Enthält geparste LTspice-Simulationsdaten."""
    source_file: str
    signals: pd.DataFrame          # Spalten = Signalnamen, Index = Zeit [s]
    metadata: dict = field(default_factory=dict)

    def signal_names(self) -> list[str]:
        return list(self.signals.columns)
