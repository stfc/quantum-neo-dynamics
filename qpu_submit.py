from typing import List
import os
import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
from qiskit import generate_preset_pass_manager, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_circuit_layout
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, EstimatorOptions, EstimatorV2

from quantum_neo_dynamics.paths import HARDWARE_EXPERIMENTS_JOBDATA, IBM_CREDENTIALS_FILE


def qpu_submit(qcs:List[QuantumCircuit], observables, backend_str="ibm_fez", tag=None, shots=100_000, noise_factors=None, mode="batch", num_randomizations="auto", job_type="zne", backend_credentials=None):
    provider = QiskitRuntimeService(channel=backend_credentials["channel"],
                                    instance=backend_credentials["instance"]
                                    )
    print(f"Provider backends: {provider.backends()}")
    backend = provider.backend(backend_str)
    props = backend.properties()
    #eplg//lf or let qiskit handlke it?
    initial_layout = None
    for qlist in props._data['general_qlists']:
        if qlist['name'] == "lf_{}".format(qcs[0].num_qubits):
            initial_layout = qlist['qubits']
            break
    pm = generate_preset_pass_manager(backend=backend, seed_transpiler=19, initial_layout=initial_layout)
    isa_qcs = pm.run(qcs)

    obs = []
    jobs = []
    for i, isa_qc in enumerate(isa_qcs):
        assert abs(qcs[i].depth(lambda x: len(x.qubits) > 1) - isa_qc.depth(lambda x: len(x.qubits) > 1)) <= 1, "our transpilation with fixed initial layout should not make things worse."
        layout = isa_qc.layout
        isa_observables = [observable.apply_layout(layout) for observable in observables]
        obs.append(isa_observables)

    if mode == "batch":
        with Batch(backend=backend) as batch:

            options = EstimatorOptions(default_shots=shots, resilience_level=0)
            # PT
            options.twirling.enable_gates = True
            options.twirling.num_randomizations = num_randomizations
            options.twirling.shots_per_randomization = "auto"
            if shots >= 100_000:
                # otherwise we have 100_000/64 many randomizations which seems excessive
                options.twirling.shots_per_randomization = 500

            # TREX
            options.resilience.measure_mitigation = True

            # ZNE
            if job_type == "zne":
                options.resilience.zne_mitigation = True
                if noise_factors is None:
                    options.resilience.zne.noise_factors = list(np.linspace(1.0, 4, 10))
                else:
                    options.resilience.zne.noise_factors = noise_factors
                options.resilience.zne.extrapolator = ("exponential", "linear", "polynomial_degree_2")
            elif job_type == "single":
                options.resilience.zne_mitigation = False

            # QOL
            # Large payload, maybe don't set
            # options.environment.log_level = "DEBUG"
            estimator = EstimatorV2(mode=batch, options=options)

            for i, isa_qc in enumerate(isa_qcs):
                if tag is None:
                    estimator.options.environment.job_tags = ["Proton_{}".format(i)]
                else:
                    estimator.options.environment.job_tags = tag
                job = estimator.run([(isa_qc, obs[i])])

                # Print correspondence between job ID and circuit name
                job_id = job.job_id()
                print("job for isa_qc=", i, "with id=", job_id)
                jobs.append((i,job_id))
    elif mode == "job":
        options = EstimatorOptions(default_shots=shots, resilience_level=0)
        # PT
        options.twirling.enable_gates = True
        options.twirling.num_randomizations = num_randomizations # set to 300
        options.twirling.shots_per_randomization = "auto"
        if shots >= 100_000:
            # otherwise we have 100_000/64 many randomizations which seems excessive
            options.twirling.shots_per_randomization = 500

        # TREX
        options.resilience.measure_mitigation = True

        # ZNE
        if job_type == "zne":
            options.resilience.zne_mitigation = True
            if noise_factors is None:
                options.resilience.zne.noise_factors = list(np.linspace(1.0, 4, 10))
            else:
                options.resilience.zne.noise_factors = noise_factors
            options.resilience.zne.extrapolator = ("exponential", "linear", "polynomial_degree_2")
        elif job_type == "single":
            options.resilience.zne_mitigation = False

        estimator = EstimatorV2(mode=backend, options=options)

        for i, isa_qc in enumerate(isa_qcs):
            if tag is None:
                estimator.options.environment.job_tags = ["Proton_{}".format(i)]
            else:
                estimator.options.environment.job_tags = tag
            job = estimator.run([(isa_qc, obs[i])])

            # Print correspondence between job ID and circuit name
            job_id = job.job_id()
            print("job for isa_qc=", i, "with id=", job_id)
            jobs.append((i,job_id))


    return jobs, isa_qcs, backend




shots = 2000
noise_factors = list(np.linspace(1.0 , 4, 5))
backend_str = "ibm_pittsburgh"
num_randomizations = 300

from quantum_neo_dynamics.decoder import load_circuits, load_hamiltonians



method = "aqc"
approximation = "low"
state = "300"
mode = "job" # job/batch
job_type = "zne" # single/zne
if job_type == "single":
    noise_factors = None

circuits = load_circuits(circuit_type=method)
circuit = circuits[f"{method}-{approximation}-{state}.qpy"][0]



# debugging
debug = False
if debug:
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(18)
    circuit.h(0)

hamiltonians = load_hamiltonians()
hamiltonian = hamiltonians[state]
commuting_groups = len(hamiltonian.group_commuting(qubit_wise=True))

# load credentials
with open(IBM_CREDENTIALS_FILE, 'r') as f:
    credentials = json.load(f)

backend_credentials = credentials.get(backend_str)
if backend_credentials is None:
    raise ValueError(f"No credentials found for backend: {backend_str}")

jobs, isa_qcs, backend = qpu_submit([circuit], [hamiltonian], backend_str=backend_str, shots=shots, 
                                    noise_factors=noise_factors, mode=mode, num_randomizations=num_randomizations, 
                                    job_type=job_type, backend_credentials=backend_credentials)


# Extract the job ID
if len(jobs) != 1:
    print("Warning: Expected exactly one job, but got", len(jobs))

job_id = jobs[0][1]
# jobid 
print("Submitted job ID:", job_id)

# Create figures directory and save plots
figures_dir = f"figures/zne_hardware/{job_id}"
os.makedirs(figures_dir, exist_ok=True)
print(f"Saving figures to {figures_dir}")

for i, isa_qc in enumerate(isa_qcs):
    plot_circuit_layout(isa_qc, backend, view="virtual")
    plt.savefig(f"{figures_dir}/layout_virtual_{i}.pdf")
    plt.close()
    
    plot_circuit_layout(isa_qc, backend, view="physical")
    plt.savefig(f"{figures_dir}/layout_physical_{i}.pdf")
    plt.close()
    
    isa_qc.draw("mpl", idle_wires=False, fold=-1)
    plt.savefig(f"{figures_dir}/isa_transpiled_{i}.pdf")
    plt.close()

# Create the record
record = {
    "job_id": job_id,
    "method": method,
    "approximation": approximation,
    "state": state,
    "shots": shots,
    "num_randomizations": num_randomizations,
    "job_type": job_type,
    "noise_factors": json.dumps(noise_factors) if noise_factors is not None else "None",
    "commuting_groups": commuting_groups,
    "experiments": commuting_groups * len(noise_factors) * shots if noise_factors is not None else commuting_groups * shots,
    "backend_str": backend_str,
    "mode": mode,
    "status": "submitted"
}
# create a DataFrame from the record
df = pd.DataFrame([record])

if os.path.exists(HARDWARE_EXPERIMENTS_JOBDATA):
    df.to_csv(HARDWARE_EXPERIMENTS_JOBDATA, mode='a', header=False, index=False)
else:
    df.to_csv(HARDWARE_EXPERIMENTS_JOBDATA, mode='w', header=True, index=False)