import json
from json import JSONDecoder
from typing import Any


from qiskit.quantum_info import SparsePauliOp

from quantum_neo_dynamics.paths import *

class ComplexDecoder(JSONDecoder):
    """JSON decoder for complex numbers."""

    def __init__(self) -> None:
        """Initialize decoder."""
        JSONDecoder.__init__(self, object_hook=ComplexDecoder.from_dict)

    @staticmethod
    def from_dict(dct: dict[str, Any]) -> complex | Any:  # noqa: ANN401
        """Decoding method.

        Args:
            dct: dictionary to decode.

        Returns:
            A complex number.
        """
        if "__complex__" in dct:
            return complex(dct["__complex__"][0], dct["__complex__"][1])
        return dct
    
def get_hamiltonian(filename: str) -> SparsePauliOp:
    """
    Create a SparsePauliOp from a JSON file containing a Hamiltonian.

    Args:
        filename (str): Path to the JSON file containing the Hamiltonian data.

    Returns:
        SparsePauliOp: The Hamiltonian as a SparsePauliOp object.
    """
    # Open and read the JSON file specified by filename
    with open(filename, "r") as f:
        # Load the Hamiltonian data using a custom decoder for complex numbers
        hamiltonian = json.load(f, cls=ComplexDecoder)

    # Extract labels and coefficients from the Hamiltonian data
    labels, coeffs = zip(*hamiltonian.items())

    # Reverse each Pauli label to conform to Qiskit convention
    labels = tuple(label[::-1] for label in labels)

    # Create and return a SparsePauliOp object using labels and coefficients
    hamiltonian_op = SparsePauliOp(labels, coeffs)

    return hamiltonian_op

def load_hamiltonians():

    paths = {
        "003": f"{HAM_DIR}/hamiltonian_003.json",
        "012": f"{HAM_DIR}/hamiltonian_012.json",
        "021": f"{HAM_DIR}/hamiltonian_021.json",
        "030": f"{HAM_DIR}/hamiltonian_030.json",
        "120": f"{HAM_DIR}/hamiltonian_120.json",
        "210": f"{HAM_DIR}/hamiltonian_210.json",
        "300": f"{HAM_DIR}/hamiltonian_300.json",
    }

    return {label: get_hamiltonian(path) for label, path in paths.items()}

def load_circuits(circuit_type: str):
    """
    Load AQC circuits from the specified directory.

    Returns:
        dict: A dictionary where keys are circuit names and values are lists of circuits.
    """
    
    from qiskit import qpy
    import os
    circ_dir = f"{CIRCUITS_DIR}/{circuit_type}"
    circuits = {}
    for filename in os.listdir(circ_dir):
        if filename.endswith(".qpy"):
            with open(os.path.join(circ_dir, filename), "rb") as f:
                circ = qpy.load(f)
                circuits[filename] = circ
    return circuits

def double_print(*args, file=None, **kwargs):
    if "file" in kwargs:
        kwargs.pop("file")
    print(*args, **kwargs)
    print(*args, **kwargs, file=file)

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def get_transpiled_circuit(circ, target):
    if target == None:
        pass_manager = generate_preset_pass_manager(optimization_level=3)
    else:
        pass_manager = generate_preset_pass_manager(target=backend.target_history(datetime=target_time), optimization_level=3)
    transpiled_circuit = pass_manager.run(circ)
    return transpiled_circuit