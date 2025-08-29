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

from quantum_neo_dynamics.paths import HARDWARE_EXPERIMENTS_JOBDATA


def qpu_submit(qcs:List[QuantumCircuit], observables, backend_str="ibm_fez", tag=None, shots=100_000, noise_factors=None, mode="batch"):
    provider = QiskitRuntimeService()
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
        plot_circuit_layout(isa_qc, backend, view="virtual")
        plt.savefig(f"layout_virtual_{i}.pdf")
        plot_circuit_layout(isa_qc, backend, view="physical")
        plt.savefig(f"layout_physical_{i}.pdf")
        isa_qc.draw("mpl", idle_wires=False, fold=-1)
        plt.savefig(f"isa_transpiled_{i}.pdf")

        isa_observables = [observable.apply_layout(layout) for observable in observables]
        obs.append(isa_observables)

    if mode == "batch":
        with Batch(backend=backend) as batch:

            options = EstimatorOptions(default_shots=shots, resilience_level=0)
            # PT
            options.twirling.enable_gates = True
            options.twirling.num_randomizations = "auto"
            options.twirling.shots_per_randomization = "auto"
            if shots >= 100_000:
                # otherwise we have 100_000/64 many randomizations which seems excessive
                options.twirling.shots_per_randomization = 500

            # TREX
            options.resilience.measure_mitigation = True

            # ZNE
            options.resilience.zne_mitigation = True
            if noise_factors is None:
                options.resilience.zne.noise_factors = list(np.linspace(1.0, 4, 10))
            else:
                options.resilience.zne.noise_factors = noise_factors
            options.resilience.zne.extrapolator = ("exponential", "linear", "polynomial_degree_2")

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
        options.twirling.num_randomizations = "auto"
        options.twirling.shots_per_randomization = "auto"
        if shots >= 100_000:
            # otherwise we have 100_000/64 many randomizations which seems excessive
            options.twirling.shots_per_randomization = 500

        # TREX
        options.resilience.measure_mitigation = True

        # ZNE
        options.resilience.zne_mitigation = True
        if noise_factors is None:
            options.resilience.zne.noise_factors = list(np.linspace(1.0, 4, 10))
        else:
            options.resilience.zne.noise_factors = noise_factors
        options.resilience.zne.extrapolator = ("exponential", "linear", "polynomial_degree_2")

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


    return jobs




shots = 2_000
noise_factors = list(np.linspace(1.0 , float(7/3), 5))
backend_str = "ibm_pittsburgh"

from quantum_neo_dynamics.decoder import load_circuits, load_hamiltonians



method = "aqc"
approximation = "low"
state = "030"
mode = "job" # job/batch

circuits = load_circuits(circuit_type=method)
circuit = circuits[f"{method}-{approximation}-{state}.qpy"][0]

hamiltonians = load_hamiltonians()
hamiltonian = hamiltonians[state]
commuting_groups = len(hamiltonian.group_commuting(qubit_wise=True))
jobs = qpu_submit([circuit], [hamiltonian], backend_str=backend_str, shots=shots, noise_factors=noise_factors, mode=mode)


# Extract the job ID
if len(jobs) != 1:
    print("Warning: Expected exactly one job, but got", len(jobs))

job_id = jobs[0][1]

# Create the record
record = {
    "job_id": job_id,
    "method": method,
    "approximation": approximation,
    "state": state,
    "shots": shots,
    "noise_factors": json.dumps(noise_factors),
    "commuting_groups": commuting_groups,
    "experiments": commuting_groups * len(noise_factors) * shots,
    "backend_str": backend_str,
    "mode": mode
}
# create a DataFrame from the record
df = pd.DataFrame([record])

if os.path.exists(HARDWARE_EXPERIMENTS_JOBDATA):
    df.to_csv(HARDWARE_EXPERIMENTS_JOBDATA, mode='a', header=False, index=False)
else:
    df.to_csv(HARDWARE_EXPERIMENTS_JOBDATA, mode='w', header=True, index=False)



