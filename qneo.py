import argparse
import json, pickle
from datetime import datetime
from mitiq import zne


from qiskit_aer.noise import NoiseModel
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from quantum_neo_dynamics.decoder import load_circuits, load_hamiltonians, double_print, get_transpiled_circuit
from quantum_neo_dynamics.paths import LOG_DIR


parser = argparse.ArgumentParser(description="Load and simulate quantum circuits.")

parser.add_argument('-s', '--state', choices=['300', '210','120','030','021','012','003'], required=True, type=str, help=(
    "State of the system defined as LMR. For method=vqe, supported systems are ['300', '030', '003']. For method=aqc and"   + 
    "approximation=high, supported systems are ['300', '030']. For method=aqc and approximation=low, supported systems "  +
    "are ['300', '210','120','030','021','012','003']"
))

parser.add_argument('-m', '--method', type=str, required=True, choices=["aqc", "vqe"], help=(
        "Method to use for the circuits. Available choices: ['aqc', 'vqe']. "
        ))

parser.add_argument('-a', '--approximation', type=str, required=True, choices=["low", "high", "shallow", "deep"], help=(
        "Approximation level for the circuits. Available choices: ['low', 'high', 'shallow', 'deep']. "
        "Note: 'low' and 'high' are for AQC circuits, while 'shallow' and 'deep' are for VQE circuits."
    ))     

parser.add_argument('-b', '--backend', type=int, default=0, choices=[0, 1, 2], help=(
        "Backend for execution, related to the choice of primitive and library used. Available choices: [0, 1, 2]. " +
        "Detailed description: " +
        "[0] (Aer statevector) uses a pure linear algebra approach with the primitive `qiskit_aer.primitives.EstimatorV2`; " + # primitive statevector does not support idle qubit trimming
        "[1] (noisy Aer statevector) same as (0) with realistic noise models."+
        "[2] (hardware) hardware experiments on ibm_fez. Not yet implemented."
))

parser.add_argument('-t', '--target', help="Target to construct the noise model. Choose from the targets files stored in $TARGETS_DIR.", type=str, required=False)

parser.add_argument('-p', '--props', help="Backend properties JSON to construct the noise model. Choose from the targets files stored in $BACKENDS_DIR.", type=str, required=False)

parser.add_argument('-nsf', '--noisescalefactor', type=float, required=False, help="Scale factor >=1 for gate unfolding. ")

parser.add_argument('-ns', '--num-shots', help="Number of shots.", type=int, default=1_000)

args = parser.parse_args()

# user variables
METHOD = args.method
APPROXIMATION = args.approximation
STATE = args.state
BACKEND = args.backend
TARGET = args.target
PROPS = args.props
NOISESCALEFACTOR = args.noisescalefactor
NUM_SHOTS = args.num_shots


if BACKEND == 1:
    from quantum_neo_dynamics.paths import TARGETS_DIR, BACKENDS_DIR


    TARGET = f"{TARGETS_DIR}/{TARGET}"
    with open(TARGET, 'rb') as target_file:
        target = pickle.load(target_file)
        

    PROPS = f"{BACKENDS_DIR}/{PROPS}"
    with open(PROPS, 'r') as props_file:
        backend_properties = BackendProperties.from_dict(
            json.load(props_file)
        )
    noise_model = NoiseModel.from_backend_properties(backend_properties)


execution_modes = ["Aer statevector",
                   "noisy Aer statevector",
                   "hardware"]

estimator_type = execution_modes[BACKEND]

time_start = datetime.now()

match estimator_type:
    case "Aer statevector":
        from qiskit_aer.primitives import EstimatorV2

        estimator = EstimatorV2()

    case "noisy Aer statevector":
        from qiskit_aer.primitives import EstimatorV2
    
        backend_options = {
            "noise_model": noise_model,
            "shots": NUM_SHOTS,
            "device": "CPU",
            "max_parallel_threads": 0,  # set to the number of CPU cores
            "max_parallel_experiments": 1,
            "max_parallel_shots": 0,  # set to max_parallel_threads
            "statevector_parallel_threshold": 10 # parallelise on or above 10 qubits
        }
        estimator = EstimatorV2(options=dict(backend_options=backend_options))

    case "hardware":
        raise NotImplementedError
        


# assert if method aqc then approximation low or high
if METHOD == "aqc":
    assert APPROXIMATION in ["low", "high"], "Approximation must be 'low' or 'high' for AQC circuits."
elif METHOD == "vqe":
    assert APPROXIMATION in ["shallow", "deep"], "Approximation must be 'shallow' or 'deep' for VQE circuits."

if METHOD == "aqc" and APPROXIMATION == "low":
    assert STATE in ["003", "012", "021", "030", "120", "210", "300"], f"State must be one of ['003', '012', '021', '030', '120', '210', '300'] for AQC circuits, got {STATE}."
elif METHOD == "aqc" and APPROXIMATION == "high":
    assert STATE in ["030", "300"], f"State must be one of ['030', '300'] for AQC circuits, got {STATE}."
elif METHOD == "vqe":
    assert STATE in ["003", "030", "300"], f"State must be one of ['003', '030', '300'] for VQE circuits, got {STATE}."

if NOISESCALEFACTOR:
    assert NOISESCALEFACTOR >= 1.00, f"Noise scale factor must be greater or equal to 1."

# load circuits and Hamiltonians
circuits = load_circuits(circuit_type=METHOD)
circuit = circuits[f"{METHOD}-{APPROXIMATION}-{STATE}.qpy"][0]

hamiltonians = load_hamiltonians()
hamiltonian = hamiltonians[STATE]

# transpile
service = QiskitRuntimeService()
if BACKEND == 0 or BACKEND ==2:
    device = "ibm_fez"  ###TODO:generalise device
    backend = service.backend(device)
    pass_manager = generate_preset_pass_manager(target=backend.target, optimization_level=3, seed_transpiler=42)
    transpiled_circuit = pass_manager.run(circuit)
    transpiled_hamiltonian = hamiltonian.apply_layout(transpiled_circuit.layout)

elif BACKEND == 1:
    pass_manager = generate_preset_pass_manager(target=target, optimization_level=3, seed_transpiler=42)
    intermediate_circuit = pass_manager.run(circuit)
    intermediate_hamiltonian = hamiltonian.apply_layout(intermediate_circuit.layout)

    zne_circuit = zne.scaling.fold_gates_at_random(intermediate_circuit, NOISESCALEFACTOR, seed=42)
    pass_manager_zne = generate_preset_pass_manager(target=target, optimization_level=0)
    transpiled_circuit = pass_manager_zne.run(zne_circuit)
    transpiled_hamiltonian = intermediate_hamiltonian.apply_layout(transpiled_circuit.layout)

pub = (transpiled_circuit, transpiled_hamiltonian )


# run job
output_log_filename = f"{LOG_DIR}/{METHOD}-{APPROXIMATION}-{STATE}-{time_start}.txt"
output_log = open(output_log_filename, 'w', buffering=1)

double_print(f"method: {METHOD}", file=output_log)
double_print(f"approximation: {APPROXIMATION}", file=output_log)
double_print(f"state: {STATE}", file=output_log)
double_print(f"backend: {estimator_type}", file=output_log)

if BACKEND !=0:
    double_print(f"num_shots: {NUM_SHOTS}", file=output_log)

if BACKEND == 1:
    double_print(f"target: {TARGET}", file=output_log)
    double_print(f"props: {PROPS}", file=output_log)
    double_print(f"noisescalefactor: {NOISESCALEFACTOR}", file=output_log) 

job = estimator.run([pub])

double_print(f"time_start: {time_start.strftime('%Y %m %d-%I %M %S %p')}", file=output_log)

if BACKEND == 2:
    double_print(f"# IBM job ID: {job.job_id()}", file=output_log)
else:
    energy = float(job.result()._pub_results[0].data.evs)
    double_print(f"\nenergy: {energy:.8f}\n", file=output_log)
    double_print(f"time_end: {datetime.now().strftime('%Y %m %d-%I %M %S %p')}", file=output_log)
    double_print(f"time_elapsed: {datetime.now() - time_start}", file=output_log)
output_log.close()



