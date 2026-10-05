import pandas as pd
from .encoding_strategy import encoding_strategy
from .result_getters import result_getter


def engineer_features(
    data: pd.DataFrame,
    encoding_stratetgy,
    result_getter,
    return_circuit : bool = False
) -> pd.DataFrame:
    """
    Perform quantum feature engineering on classical tabular data.

    This function acts as a high-level orchestration pipeline that transforms
    classical data into quantum-derived features using an encoding strategy
    and a result extraction backend.

    The pipeline consists of two main stages:

    1. Encoding Stage
       - The input dataframe is passed into an encoding strategy.
       - The encoding strategy converts each row into a Primitive Unitary Block (PUB) of the form:
             (QuantumCircuit, parameter_values)

    2. Evaluation Stage
       - The generated PUBs are passed to a result getter.
       - The result getter evaluates quantum circuits (either exactly or via
         a runtime backend such as Qiskit Estimator).
       - Outputs are returned as a structured feature matrix.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing classical features.

        - Rows represent individual samples/observations.
        - Columns represent classical feature dimensions.

    encoding_stratetgy : object
        Encoding strategy object responsible for converting classical data
        into quantum circuit inputs.

        Must implement:
            generate_pubs(data) -> List[tuple[QuantumCircuit, list[float]]]

    result_getter : object
        Result extraction object responsible for evaluating PUBs and
        returning feature representations.

        Must implement:
            get_results(pubs) -> pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Quantum-engineered feature matrix.

        - Rows correspond to input samples.
        - Columns correspond to quantum-derived features.
    """

    pubs = encoding_stratetgy.generate_pubs(data)
    results = result_getter.get_results(pubs)
    if return_circuit:
        return results, pubs[0][0]
    else:
        return results