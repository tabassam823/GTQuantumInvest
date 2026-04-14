import numpy as np
import pandas as pd
import os
import pennylane as qml
from run_spsa_test import run_spsa_test

def find_optimal_lr_spsa(H, n_qubits, curr_date, ne_bitstring=None, K=2, test_iters=30):
    """Mencari Learning Rate (a) optimal dan mengekspor ke CSV dengan tanggal."""
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def cost_circuit(params):
        depth = 1 
        w = params.reshape((depth + 1, n_qubits, 2))
        for layer in range(depth + 1):
            for q in range(n_qubits):
                qml.RY(w[layer, q, 0], wires=q)
                qml.RZ(w[layer, q, 1], wires=q)
            if layer < depth:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
        return qml.expval(H)

    n_params = n_qubits * 2 * (1 + 1)
    rng = np.random.default_rng(42)
    
    if ne_bitstring is not None:
        init_w = np.zeros((2, n_qubits, 2))
        for q in range(n_qubits):
            bit = int(ne_bitstring[q])
            if bit == 1: init_w[0, q, 0] = np.pi
            else: init_w[0, q, 0] = 0.0
        init_p = init_w.flatten() + rng.uniform(-0.1, 0.1, n_params)
    else:
        init_p = rng.uniform(0, 2 * np.pi, n_params)
        
    test_a_values = np.logspace(-3, 0, 15) 
    final_energies = []
    for a_test in test_a_values:
        energy = run_spsa_test(cost_circuit, n_params, init_params=init_p, a_base=a_test, max_iters=test_iters)
        final_energies.append(energy)
        
    # --- EKSPOR HASIL LR FINDER (Poin 3 & 5) ---
    lr_df = pd.DataFrame({
        'Date': [curr_date.date()] * len(test_a_values),
        'LearningRate': test_a_values, 
        'Energy': final_energies
    })
    lr_df.to_csv('hasil_pencarian_lr.csv', mode='a', header=not os.path.exists('hasil_pencarian_lr.csv'), index=False)

    best_idx = np.argmin(final_energies)
    return test_a_values[best_idx], test_a_values, final_energies
