import numpy as np
import networkx as nx
import rustworkx
import matplotlib.pyplot as plt
import copy
from qiskit_addon_cutting.instructions import Move
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeMarrakesh, FakeFez
from qiskit_ibm_runtime import SamplerV2, EstimatorV2
from qiskit_aer.primitives.sampler_v2 import SamplerV2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.debug_tools import Neat
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Batch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.providers.backend import BackendV2
from typing import Dict, List
from qiskit.transpiler import Target, Layout
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import (
    SXGate, XGate, Measure,
    CZGate, RZGate, SwapGate
)
from qiskit_addon_cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)

from qiskit_addon_cutting import (
    cut_wires,
    expand_observables,
    partition_problem,
    generate_cutting_experiments,
)
from qiskit_addon_cutting import reconstruct_expectation_values


def run_hardware_inspired_cutting(circuit, backend, cut_budget, observable, z_score):
    """
    Hardware-Inspired Cutting (HIC)
    
    Input:
        - circuit: The quantum circuit to be optimized.
        - backend: The backend on which the circuit will be executed.
        - cut_budget: The budget for cutting the circuit.
        - observable: The observable to be measured.
        - z_score: The z-score threshold for identifying outlier qubits and edges in the hardware.

    Returns:
        {
            "expectation_value": float,
            "num_cuts": int,
            "weighted_average": float,
            "ideal_expval": float
        }
    """

    # ======================================================
    # 1. Identify bad qubits
    # ======================================================
    def identify_bad_qubits(z_score:float):
        """
            If some qubits are really bad, and statistical outliers, it is useful
            to remove those qubits
        """
        if 'sx' in backend.configuration().basis_gates:
            basis_1q_gate = 'sx'
        elif 'x' in backend.configuration().basis_gates:
            basis_1q_gate = 'x'
        else:
            basis_1q_gate = 'rz'

        gate_1q_error = {}
        spam_error = {}
        
        for qubit in range(backend.num_qubits):
            spam_error[qubit] = backend.properties().readout_error(qubit)
            gate_1q_error[qubit] = backend.properties().gate_error(basis_1q_gate, [qubit])

        
        if max(list(gate_1q_error.values())) == 1:
            screened_gate_1q_error = {}
            for edge, val in gate_1q_error.items():
                if val < 1:
                    screened_gate_1q_error[edge] = val
            mean_score_1qubit = np.mean(list(screened_gate_1q_error.values()))
            std_score_1qubit = np.std(list(screened_gate_1q_error.values()))
        else:
            mean_score_1qubit = np.mean(list(gate_1q_error.values()))
            std_score_1qubit = np.std(list(gate_1q_error.values()))

        

        mean_score_spam = np.mean(list(spam_error.values()))
        std_score_spam = np.std(list(spam_error.values()))

        retained = {}
        outliers = {}

        for qubit, score in spam_error.items():
            if (score-mean_score_spam)/std_score_spam > z_score :
                outliers[qubit] = score
            else:
                retained[qubit] = score
        
        for qubit, score in gate_1q_error.items():
            if (score-mean_score_1qubit)/std_score_1qubit > z_score :
                outliers[qubit] = score
            else:
                retained[qubit] = score

        return outliers, retained

    # ======================================================
    # 2. Identify bad edges
    # ======================================================
    def identify_bad_edges(z_score:float):
        """
        If some edges are really bad, and statistical outliers, it is useful
        to remove those edges
        """
        if 'cx' in backend.configuration().basis_gates:
            basis_2q_gate = 'cx'
        elif 'ecr' in backend.configuration().basis_gates:
            basis_2q_gate = 'ecr'
        else:
            basis_2q_gate = 'cz'

        gate_2q_error = {}

        for qubit1 in range(backend.num_qubits):
            for qubit2 in range(backend.num_qubits):
                if [qubit1, qubit2] in backend.configuration().coupling_map:
                    gate_2q_error[(qubit1, qubit2)] = backend.properties().gate_error(basis_2q_gate, [qubit1, qubit2])

        if max(list(gate_2q_error.values())) == 1:
            screened_gate_2q_error = {}
            for edge, val in gate_2q_error.items():
                if val < 1:
                    screened_gate_2q_error[edge] = val
            mean_score = np.mean(list(screened_gate_2q_error.values()))
            std_score = np.std(list(screened_gate_2q_error.values()))
        else:
            mean_score = np.mean(list(gate_2q_error.values()))
            std_score = np.std(list(gate_2q_error.values()))

        retained = {}
        outliers = {}

        for edge, score in gate_2q_error.items():
            if (score-mean_score)/std_score > z_score:
                outliers[edge] = score
            else:
                retained[edge] = score

        return outliers, retained

    # ======================================================
    # 3. Create punctured coupling map
    # ======================================================
    def create_punctured_coupling_map(spam_outliers: dict, edge_outliers: dict):
        """
        Create punched coupling map by removing spam and edge outliers
        """
        coupling_map = backend.configuration().coupling_map
        punctured_coupling_map = []

        bad_qubits = list(spam_outliers.keys())
        bad_edges = list(edge_outliers.keys())

        for edge in coupling_map:
            if edge not in bad_edges and edge[0] not in bad_qubits and edge[1] not in bad_qubits:
                punctured_coupling_map.append(edge)  

        return punctured_coupling_map

    # ======================================================
    # 4. Connected components = low-noise islands
    # ======================================================
    def get_connected_components(punctured_coupling_map: tuple) -> dict:
        """
        Create a dictionary with the connected components
        """
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(punctured_coupling_map)

        coupling_map = backend.configuration().coupling_map
        component_graphs = {}

        for component in list(nx.connected_components(temp_graph)):
            c_graph = []
            for u in component:
                for v in component:
                    if [u,v] in coupling_map:
                        c_graph.append([u,v])
            component_graphs[len(component)] = c_graph

        return component_graphs

    # ======================================================
    # Run noise profiling
    # ======================================================
    spam_outliers, _ = identify_bad_qubits(z_score)
    edge_outliers, _ = identify_bad_edges(z_score)
    punctured_coupling_map = create_punctured_coupling_map(spam_outliers, edge_outliers)
    component_graphs = get_connected_components(punctured_coupling_map)

    print("Component graph sizes are : ", component_graphs.keys())

    # ======================================================
    # 5. Transpilation and Clifford Conversion
    # ======================================================
    pm = generate_preset_pass_manager(
        optimization_level=3, 
        basis_gates=backend.configuration().basis_gates,
        seed_transpiler=1
    )

    # Run transpiler to map to basis gates
    circuit = pm.run(circuit)
    analyzer = Neat(backend) 

    # Convert the circuit to Clifford pubs for analysis
    clifford_pubs = analyzer.to_clifford([(circuit, observable)])

    circuit = clifford_pubs[0].circuit

    # Ensure it’s a plain QuantumCircuit
    if not isinstance(circuit, QuantumCircuit):
        circuit = QuantumCircuit.from_instructions(circuit)

    # Determine subcircuit size search range based on islands
    min_size = circuit.num_qubits // 2
    if len(list(component_graphs.keys())) != 1 and min(list(component_graphs.keys())) > min_size:
        min_size = min(list(component_graphs.keys()))

    potential_subcircuit_sizes = np.arange(
                                        min_size, 
                                        max(list(component_graphs.keys())) + 1
                                        )[::-1]
    
    # ======================================================
    # 7. Ideal Baseline Calculation
    # ======================================================
    # Calculate the noise-free expectation value for benchmarking
    estimator_ideal = EstimatorV2(mode=AerSimulator())
    result_ideal = (
        estimator_ideal.run([(circuit, observable)]).result()[0].data.evs.item()
    )
    print("The ideal expectation value is : ", result_ideal)

    # ======================================================
    # 8. Scoring and Layout Helper Functions
    # ======================================================
    def calculate_weighted_avg(layouts: dict) -> float:
        """Calculates the weighted average error score across total number of qubits in the circuit."""
        total_qubits = circuit.num_qubits
        weighted_sum = sum((len(v[0]) / total_qubits) * v[1] for v in layouts.values())
        return weighted_sum

    def _qid(q):
        """Helper to get the index of a qubit."""
        return q if isinstance(q, int) else q._index

    def cost_func(circ, layouts, backend):
        """
        Calculates a fidelity-based cost for a circuit layout based on 
        gate and readout errors from the target backend.
        """
        target = backend.target
        out = []
        two_q_gates = [g for g in ("cx", "cz", "ecr") if g in target]

        for layout in layouts:
            fid = 1.0
            local_to_global = {
                _qid(vq): _qid(pq) 
                for pq, vq in layout.get_physical_bits().items()
            }

            for inst in circ.data:
                name = inst.operation.name
                local_qubits = [_qid(q) for q in inst.qubits]
                try:
                    global_qubits = tuple(local_to_global[q] for q in local_qubits)
                except KeyError:
                    continue

                # Calculate cumulative fidelity based on gate errors
                if name in two_q_gates and len(global_qubits) == 2:
                    props = target[name].get(global_qubits) or target[name].get(global_qubits[::-1])
                    if props: fid *= (1 - props.error)
                elif name in ("sx", "x"):
                    props = target[name].get((global_qubits[0],))
                    if props: fid *= (1 - props.error)
                elif name == "measure":
                    props = target["measure"].get((global_qubits[0],))
                    if props: fid *= (1 - props.error)

            out.append((layout, 1 - fid)) # Returns (layout, error_rate)
        return out

    # ======================================================
    # 9. Hardware Island Emulation
    # ======================================================
    def create_component_backend_with_real_noise(backend, edgelist):
        """
        Creates a virtual backend representing a specific connected 
        component (island) of the real hardware.
        """
        real_target = backend.target
        component_qubits = sorted({q for a, b in edgelist for q in (a, b)})
        n = len(component_qubits)
        global_to_local = {q: i for i, q in enumerate(component_qubits)}

        component_target = Target(num_qubits=n)
        component_target.qubit_properties = [
            copy.deepcopy(real_target.qubit_properties[q]) for q in component_qubits
        ]
        
        # Map native gates and their real error properties to the island
        sx_props, x_props, meas_props, cz_props = {}, {}, {}, {}
        for gq in component_qubits:
            lq = global_to_local[gq]
            if "sx" in real_target and (gq,) in real_target["sx"]:
                sx_props[(lq,)] = real_target["sx"][(gq,)]
            if "x" in real_target and (gq,) in real_target["x"]:
                x_props[(lq,)] = real_target["x"][(gq,)]
            if "measure" in real_target and (gq,) in real_target["measure"]:
                meas_props[(lq,)] = real_target["measure"][(gq,)]

        for q0, q1 in edgelist:
            if (q0, q1) in real_target["cz"]:
                cz_props[(global_to_local[q0], global_to_local[q1])] = real_target["cz"][(q0, q1)]
            elif (q1, q0) in real_target["cz"]:
                cz_props[(global_to_local[q0], global_to_local[q1])] = real_target["cz"][(q1, q0)]

        if sx_props: component_target.add_instruction(SXGate(), sx_props)
        if x_props: component_target.add_instruction(XGate(), x_props)
        if meas_props: component_target.add_instruction(Measure(), meas_props)
        if cz_props: component_target.add_instruction(CZGate(), cz_props)

        component_target.add_instruction(RZGate(0), {(i,): None for i in range(n)})
        component_target.add_instruction(SwapGate())

        return GenericBackendV2(
            num_qubits=n,
            coupling_map=[(global_to_local[a], global_to_local[b]) for a, b in edgelist],
            basis_gates=backend.configuration().basis_gates,
            noise_info=False,
        ), component_target

    # ======================================================
    # 10. Subcircuit Performance Profiling
    # ======================================================
    def collect_component_subcircuit_data(component_graphs, circuits, backend):
        """
        Evaluates every subcircuit against every hardware island to 
        find the optimal noise-aware mapping.
        """
        all_data = {}
        for size, edgelist in component_graphs.items():
            comp_backend, comp_target = create_component_backend_with_real_noise(backend, edgelist)
            pm = generate_preset_pass_manager(
                target=comp_target,
                basis_gates=backend.configuration().basis_gates,
                optimization_level=3,
            )

            component_results = {}
            for idx, ckt in enumerate(circuits):
                if ckt.num_qubits > comp_backend.num_qubits:
                    continue

                isa = pm.run(ckt)
                layout = isa.layout.final_layout if isa.layout else Layout.from_intlist(list(range(isa.num_qubits)), *isa.qregs)
                
                # Score the ISA circuit based on the REAL backend's noise target
                score = cost_func(isa, [layout], backend)[0][1]
                component_results[idx] = {"score": score, "layout": layout}

            all_data[size] = component_results
        return all_data
    
    def layout_to_physical_qubits(layout):
        """
        Returns a sorted list of physical qubit indices used by the circuit layout,
        filtering out ancilla and unused bits.
        """
        phys = []
        for phys_idx, virt_bit in layout.get_physical_bits().items():
            if virt_bit is None:
                continue
            # Skip ancilla qubits safely
            reg = getattr(virt_bit, "register", None)
            if reg is not None and reg.name == "ancilla":
                continue
            phys.append(phys_idx)
        return sorted(phys)

    def select_best_component_per_subcircuit(all_scores):
        """
        Ranks subcircuit mappings across all hardware islands and selects the 
        best physical component for each fragment based on the error score.
        """
        best = {}
        for component_size, sub_dict in all_scores.items():
            for sub_idx, data in sub_dict.items():
                score = data["score"]
                layout = data["layout"]

                # Select based on lowest score (fidelity error)
                # Tie-break with smaller component size (denser connectivity)
                if (
                    sub_idx not in best
                    or score < best[sub_idx]["score"]
                    or (
                        score == best[sub_idx]["score"]
                        and component_size < best[sub_idx]["component_size"]
                    )
                ):
                    best[sub_idx] = {
                        "component_size": component_size,
                        "score": score,
                        "sub_mapping": layout_to_physical_qubits(layout),
                    }
        return best

    # ======================================================
    # 11. Search for Optimal Cuts within Budget
    # ======================================================
    if cut_budget == -1:
        # --- Branch A: Equal Partitioning ---
        optimization_settings = OptimizationParameters(seed=111)
        device_constraints_equal_partitioning = DeviceConstraints(
            qubits_per_subcircuit=circuit.num_qubits // 2
        )
        
        cut_circuit_equal_partitioning, metadata_equal_partitioning = find_cuts(
            circuit, optimization_settings, device_constraints_equal_partitioning
        )
        
        qc_w_ancilla_equal_partitioning = cut_wires(cut_circuit_equal_partitioning)
        observables_expanded_equal_partitioning = expand_observables(
            observable.paulis, circuit, qc_w_ancilla_equal_partitioning
        )
        
        partitioned_problem_equal_partitioning = partition_problem(
            circuit=qc_w_ancilla_equal_partitioning, 
            observables=observables_expanded_equal_partitioning
        )
        
        subcircuits_equal_partitioning = partitioned_problem_equal_partitioning.subcircuits
        subobservables_equal_partitioning = {
            k: v for k, v in partitioned_problem_equal_partitioning.subobservables.items() 
            if k is not None
        }
        
        subexperiments_equal_partitioning, coefficients_equal_partitioning = generate_cutting_experiments(
            circuits=subcircuits_equal_partitioning, 
            observables=subobservables_equal_partitioning, 
            num_samples=np.inf
        )
        
        # Scoring for Equal Partitioning
        all_scores_equal_partitioning = collect_component_subcircuit_data(
            component_graphs, 
            [subexperiments_equal_partitioning[i][0] for i in subexperiments_equal_partitioning.keys()], 
            backend
        )
        
        best_assignments_final = select_best_component_per_subcircuit(all_scores_equal_partitioning)
        
        final_expts = subexperiments_equal_partitioning
        final_coeffs = coefficients_equal_partitioning
        final_obs = subobservables_equal_partitioning
        final_num_cuts = len(metadata_equal_partitioning["cuts"])
        final_weight = calculate_weighted_avg({
            idx: (info["sub_mapping"], info["score"]) 
            for idx, info in best_assignments_final.items()
        })

    else:
        # --- Branch B: Budgeted Search ---
        optimization_settings = OptimizationParameters(seed=111)
        best_weight = float("inf")
        best_val = None
        best_scores_budget = None
        best_subcircuits_budget = None
        best_coefficients_budget = None 

        for val in potential_subcircuit_sizes:
            val = int(val)
            if val >= circuit.num_qubits: continue
            
            device_constraints_budget = DeviceConstraints(qubits_per_subcircuit=val)
            cut_circuit_budget, metadata_budget = find_cuts(circuit, optimization_settings, device_constraints_budget)
            num_cuts_budget = len(metadata_budget["cuts"])

            if num_cuts_budget > cut_budget:
                print(f"Skipping size {val} because cuts ({num_cuts_budget}) exceed budget ({cut_budget})")
                continue
            else:
                qc_w_ancilla_budget = cut_wires(cut_circuit_budget)
                obs_expanded_budget = expand_observables(observable.paulis, circuit, qc_w_ancilla_budget)
                partitioned_budget = partition_problem(circuit=qc_w_ancilla_budget, observables=obs_expanded_budget)
                subcircuits_budget = partitioned_budget.subcircuits
                subobs_budget = {k: v for k, v in partitioned_budget.subobservables.items() if k is not None}
                
                subexpts_budget, coeffs_budget = generate_cutting_experiments(
                    circuits=subcircuits_budget, observables=subobs_budget, num_samples=np.inf
                )
                
                all_scores_budget = collect_component_subcircuit_data(
                    component_graphs, 
                    [subexpts_budget[i][0] for i in subexpts_budget.keys()], 
                    backend
                )

                best_assignments_tmp = select_best_component_per_subcircuit(all_scores_budget)
                w_si = calculate_weighted_avg({
                    idx: (info["sub_mapping"], info["score"]) 
                    for idx, info in best_assignments_tmp.items()
                })

                print(f"Weighted score w_s for subcircuit size {val}: {w_si}")
                
                if w_si < best_weight:
                    best_weight = w_si
                    best_val = val
                    best_scores_budget = all_scores_budget
                    best_metadata_budget = metadata_budget
                    best_subobservables_budget = subobs_budget
                    best_subexperiments_budget = subexpts_budget
                    best_coefficients_budget = coeffs_budget

        # Unify variables for final steps 
        final_expts = best_subexperiments_budget
        final_coeffs = best_coefficients_budget
        final_obs = best_subobservables_budget
        final_num_cuts = len(best_metadata_budget["cuts"])
        final_weight = best_weight
        best_assignments_final = select_best_component_per_subcircuit(best_scores_budget)

    # ======================================================
    # 12. Physical Mapping (ISA Transpilation)
    # ======================================================
    pass_managers = {}
    isa_subexperiments = {}
    isa_subobservables = {}

    for label, subexp_list in final_expts.items():
        sub_mapping = best_assignments_final[label]["sub_mapping"]

        pm_isa = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
            initial_layout=sub_mapping
        )
        pass_managers[label] = pm_isa

        isa_subexperiments[label] = [pm_isa.run(circ) for circ in subexp_list]
        isa_subobservables[label] = (
            SparsePauliOp(final_obs[label])
            .apply_layout(sub_mapping, num_qubits=backend.num_qubits)
        )

    # ======================================================
    # 13. Clifford Conversion and Noisy Simulation
    # ======================================================

    def extract_clifford_pair(pub):
        # ---- circuit ----
        if hasattr(pub, "circuit"):
            circ = pub.circuit
        elif hasattr(pub, "circuits"):
            circ = pub.circuits[0]
        else:
            raise TypeError("EstimatorPub has no circuit")

        # ---- observable ----
        obs_container = pub.observables

        # Convert ObservablesArray safely
        if hasattr(obs_container, "to_list"):
            obs_list = obs_container.to_list()
        else:
            obs_list = list(obs_container)

        # Case 1: empty observable means Identity
        if len(obs_list) == 0:
            n = circ.num_qubits
            obs = SparsePauliOp("I" * n)

        # Case 2: normal observable
        else:
            obs = obs_list[0]

        return circ, obs
    
    clifford_subexps = {}
    clifford_subobs = {}

    for label, subexp in isa_subexperiments.items():
        clifford_subexps[label] = []
        obs_mapped = isa_subobservables[label]

        for circ in subexp:
            pub = analyzer.to_clifford([(circ, obs_mapped)])[0]
            
            cliff_circ, cliff_obs = extract_clifford_pair(pub)

            if not isinstance(cliff_circ, QuantumCircuit):
                cliff_circ = QuantumCircuit.from_instructions(cliff_circ)

            clifford_subexps[label].append(cliff_circ)
        clifford_subobs[label] = cliff_obs

    # Execute trials with Noise Model
    num_trials = 1
    noise_model = NoiseModel.from_backend(backend, thermal_relaxation=False, readout_error=False)
    backend_options = {"method": "stabilizer", "noise_model": noise_model}

    reconstructed_expvals = []
    
    with Batch(backend=backend):
        sampler = SamplerV2(options={"backend_options": backend_options}, default_shots=2**12)
        
        for idx in range(num_trials):
            job_trial = {
                label: sampler.run(sub_list)
                for label, sub_list in clifford_subexps.items()
            }
            
            trial_results = {label: job.result() for label, job in job_trial.items()}
            
            expval_terms = reconstruct_expectation_values(
                trial_results,
                final_coeffs,
                final_obs
            )
            
            # Combine terms using the original global observable coeffs
            final_val = np.dot(expval_terms, observable.coeffs).real
            reconstructed_expvals.append(final_val)

    # ======================================================
    # 14. Final Return
    # ======================================================
    return {
        "expectation_value": np.mean(reconstructed_expvals),
        "num_cuts": final_num_cuts,
        "weighted_average": final_weight,
        "ideal_expval": result_ideal
    }

