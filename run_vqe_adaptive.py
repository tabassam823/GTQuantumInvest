import numpy as np
import pandas as pd
import os
import pennylane as qml

def run_vqe_adaptive(H, n_qubits, curr_date, ne_bitstring=None, K=2, max_depth=4,
                     maxiter=100, max_total_iter=2000,
                     batch_size=25, conv_window=4, conv_tol=1e-4,
                     best_a_base=0.1, seed=42):
    """
    Menjalankan VQE dengan seeding awal (warm-start) dari Nash Equilibrium.
    Mengekspor riwayat iterasi (hanya window terakhir) dan pencarian depth ke CSV.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    rng = np.random.default_rng(seed)

    # --- RESET RIWAYAT ITERASI (Poin 2) ---
    # File ini dikosongkan setiap kali window baru dimulai
    if os.path.exists('riwayat_iterasi_vqe.csv'):
        os.remove('riwayat_iterasi_vqe.csv')

    def make_circuit(depth):
        @qml.qnode(dev)
        def cost_circuit(params):
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

        @qml.qnode(dev)
        def prob_circuit(params):
            w = params.reshape((depth + 1, n_qubits, 2))
            for layer in range(depth + 1):
                for q in range(n_qubits):
                    qml.RY(w[layer, q, 0], wires=q)
                    qml.RZ(w[layer, q, 1], wires=q)
                if layer < depth:
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])
            return qml.probs(wires=range(n_qubits))

        return cost_circuit, prob_circuit

    def run_spsa(cost_circuit, n_params, init_params=None, a_base=0.1, c_base=0.1, depth_label=1):
        a, c      = a_base, c_base
        A_coeff   = maxiter * 0.1
        alpha_exp = 0.602
        gamma_exp = 0.101

        if init_params is not None:
            params = init_params.copy()
        else:
            params = rng.uniform(0, 2 * np.pi, n_params)

        energy_history = []
        total_iters    = 0
        while total_iters < max_total_iter:
            for _ in range(batch_size):
                k     = total_iters
                a_k   = a / (A_coeff + k + 1) ** alpha_exp
                c_k   = c / (k + 1)           ** gamma_exp
                delta = 2 * rng.integers(0, 2, size=n_params) - 1
                cost_plus  = float(cost_circuit(params + c_k * delta))
                cost_minus = float(cost_circuit(params - c_k * delta))
                grad       = (cost_plus - cost_minus) / (2 * c_k * delta)
                params     = params - a_k * grad
                total_iters += 1

            current_energy = float(cost_circuit(params))
            energy_history.append(current_energy)
            
            # --- EKSPOR RIWAYAT ITERASI (Poin 2) ---
            iter_df = pd.DataFrame({'Depth': [depth_label], 'Iteration': [total_iters], 'Energy': [current_energy]})
            iter_df.to_csv('riwayat_iterasi_vqe.csv', mode='a', header=not os.path.exists('riwayat_iterasi_vqe.csv'), index=False)

            if total_iters >= maxiter and len(energy_history) >= conv_window:
                E_old, E_now = energy_history[-conv_window], energy_history[-1]
                if abs(E_old - E_now) / (abs(E_old) + 1e-12) < conv_tol:
                    break
        return params, energy_history[-1], energy_history, total_iters

    best_energy    = np.inf
    best_params, best_depth, best_history = None, 1, []
    depth_energies = []
    prev_params    = None
    all_theta_data = []

    for depth in range(1, max_depth + 1):
        n_params = n_qubits * 2 * (depth + 1)
        cost_fn, prob_fn = make_circuit(depth)

        if prev_params is not None and len(prev_params) < n_params:
            init_p = np.concatenate([prev_params, rng.uniform(-0.1, 0.1, n_params - len(prev_params))])
        else:
            if ne_bitstring is not None:
                init_w = np.zeros((depth + 1, n_qubits, 2))
                for q in range(n_qubits):
                    init_w[0, q, 0] = np.pi if int(ne_bitstring[q]) == 1 else 0.0
                init_p = init_w.flatten() + rng.uniform(-0.1, 0.1, n_params)
            else:
                init_p = rng.uniform(0, 2 * np.pi, n_params)

        params, energy, e_hist, n_iters = run_spsa(cost_fn, n_params, init_params=init_p, a_base=best_a_base/depth, c_base=0.1/np.sqrt(depth), depth_label=depth)
        print(f"    [Depth {depth}] Konvergen dalam {n_iters} iterasi | E = {energy:.6f}")
        depth_energies.append((depth, energy, n_iters))
        prev_params = params

        # --- KUMPULKAN THETA (Poin 1) ---
        for idx, val in enumerate(params):
            all_theta_data.append({'Depth': depth, 'Theta_Index': idx, 'Theta_Value': val})

        if energy < best_energy:
            best_energy, best_params, best_depth, best_history = energy, params, depth, e_hist
            
    # --- EKSPOR THETA GABUNGAN (Poin 1) ---
    pd.DataFrame(all_theta_data).to_csv('theta_final_all_depths.csv', index=False)

    # --- EKSPOR DEPTH VS ENERGI (Poin 3) ---
    depth_df = pd.DataFrame(depth_energies, columns=['Depth', 'Energy', 'Iterations'])
    depth_df.insert(0, 'Date', curr_date.date())
    depth_df.to_csv('hasil_depth_vs_energi.csv', mode='a', header=not os.path.exists('hasil_depth_vs_energi.csv'), index=False)

    _, prob_fn = make_circuit(best_depth)
    probs = prob_fn(best_params)
    sorted_indices = np.argsort(probs)[::-1]
    best_bitstring = None
    for idx in sorted_indices:
        bs = format(idx, f'0{n_qubits}b')
        if bs.count('1') == K:
            best_bitstring = bs
            break
    if best_bitstring is None:
        top_k = np.argsort(probs)[-K:]
        bs_list = list('0' * n_qubits)
        for idx in top_k: bs_list[idx] = '1'
        best_bitstring = ''.join(bs_list)

    return [i for i, bit in enumerate(best_bitstring) if bit == '1'], best_depth, best_energy, best_history, depth_energies
