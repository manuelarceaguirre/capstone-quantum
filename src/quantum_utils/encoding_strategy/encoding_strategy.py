from qiskit import QuantumCircuit
from typing import List
import pandas as pd

class encoding_strategy():
    def generate_pubs(self, data : pd.DataFrame) -> list[tuple[QuantumCircuit, List[float]]]:
        pass