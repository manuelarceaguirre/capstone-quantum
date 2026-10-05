import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from typing import List, Tuple

class result_getter():
    '''Base class for result getters. The result getter object holds the information and logic to translate '''
    def get_results(self, pubs : List[Tuple[QuantumCircuit, List[float]]]) -> pd.DataFrame:
        '''Get results for the given publications.'''  
        pass