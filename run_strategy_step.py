import numpy as np
import pandas as pd
import os
from compute_endogenous_lambda import compute_endogenous_lambda
from find_nash_sbr import find_nash_sbr
from build_hamiltonian_total import build_hamiltonian_total
from find_optimal_lr_spsa import find_optimal_lr_spsa
from run_vqe_adaptive import run_vqe_adaptive

def run_strategy_step(lookback_data, tickers, curr_date, K=2, penalty_A=5.0,
                      max_depth=6, maxiter=100,
                      max_total_iter=2000, batch_size=25,
                      conv_window=4, conv_tol=1e-4):
    """
    Eksekusi satu periode pembelajaran dengan pencatatan tanggal.
    """
    n_assets = len(tickers)
    # Simple Return
    simple_rets = lookback_data.pct_change().dropna()
    # Log Return
    log_rets  = np.log(lookback_data / lookback_data.shift(1)).dropna()
    
    binary_st = (log_rets <= 0).astype(int)   # 1 = turun, 0 = naik

    # gamma disini mewakili degree of risk-aversion, yang diestimasi serupa endogenous lambda
    gamma = compute_endogenous_lambda(log_rets, tickers)
    lam = penalty_A # penalty lambda untuk pembatas kardinalitas
    
    # Menggunakan return periode (126 hari) dan covariance secara langsung tanpa NMI
    mu_simple_period = simple_rets.mean().values * 126
    mu_log_period = log_rets.mean().values * 126
    sigma_log = log_rets.std().values
    sigma_period_log = sigma_log * np.sqrt(126)

    # --- EKSPOR DATA RETURN & METRIK (Poin 2 & 3) ---
    metrics_df = pd.DataFrame({
        'Date': [curr_date.date()] * n_assets,
        'Ticker': tickers,
        'Mu_Simple_Period': mu_simple_period,
        'Mu_Log_Period': mu_log_period,
        'Sigma_Log': sigma_log,
        'Sigma_Period_Log': sigma_period_log,
        'Lambda_RiskAversion': [gamma] * n_assets
    })
    
    file_path_metrics = 'metrik_return_dan_lambda.csv'
    metrics_df.to_csv(file_path_metrics, mode='a', header=not os.path.exists(file_path_metrics), index=False)

    # Konstruksi Hamiltonian Ising (Pauli-Z) Sesuai PDF
    h_total = np.zeros(n_assets)
    J_total = np.zeros((n_assets, n_assets))

    # Gunakan return periode (126 hari) dan covariance secara langsung untuk Hamiltonian
    mu_avg_period = mu_log_period
    sigma_period_matrix = log_rets.cov().values * 126

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            J_val = (gamma * sigma_period_matrix[i, j] + 2 * lam) / 4.0
            J_total[i, j] = J_val
            J_total[j, i] = J_val
            
    for i in range(n_assets):
        sum_J_ij = 0.0
        for j in range(n_assets):
            if i != j:
                sum_J_ij += (gamma * sigma_period_matrix[i, j] + 2 * lam) / 4.0
        h_total[i] = -0.5 * ((gamma / 2.0) * sigma_period_matrix[i, i] - mu_avg_period[i] + lam * (1.0 - 2.0 * K)) - sum_J_ij

    # --- EKSPOR BIAS & INTERAKSI QUBO (Poin 3 & 4) ---
    h_df = pd.DataFrame({'Date': [curr_date.date()] * n_assets, 'Ticker': tickers, 'Bias_h': h_total})
    h_df.to_csv('bias_h_total.csv', mode='a', header=not os.path.exists('bias_h_total.csv'), index=False)
    
    J_list = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            J_list.append({'Date': curr_date.date(), 'Ticker_i': tickers[i], 'Ticker_j': tickers[j], 'Interaction_J': J_total[i, j]})
    J_df = pd.DataFrame(J_list)
    J_df.to_csv('interaksi_J_total.csv', mode='a', header=not os.path.exists('interaksi_J_total.csv'), index=False)

    # [Tugas 4] Pencarian Nash Equilibrium
    ne_bitstring, ne_energy = find_nash_sbr(h_total, J_total, curr_date, N=n_assets, K=K)
    print(f"    [Nash Eq] Bitstring: {ne_bitstring} | Energy: {ne_energy:.6f}")

    H = build_hamiltonian_total(h_total, J_total, n_assets)
    
    # [LR Finder] Mencari Learning Rate optimal untuk bulan ini
    print("    [LR Finder] Mencari Learning Rate optimal...")
    best_a, test_a_values, final_energies = find_optimal_lr_spsa(H, n_assets, curr_date, ne_bitstring=ne_bitstring, K=K, test_iters=30)
    print(f"    [LR Finder] Base Learning Rate terpilih: {best_a:.4f}")

    # Optimasi VQE dengan adaptive depth dan Base LR terkalibrasi
    selected_indices, depth_used, energy_final, best_history, depth_energies = run_vqe_adaptive(
        H, n_assets, curr_date, ne_bitstring=ne_bitstring, K=K, max_depth=max_depth, maxiter=maxiter,
        max_total_iter=max_total_iter, batch_size=batch_size,
        conv_window=conv_window, conv_tol=conv_tol, 
        best_a_base=best_a
    )

    lr_data = (test_a_values, final_energies, best_a)
    return selected_indices, depth_used, energy_final, best_history, depth_energies, lr_data, ne_bitstring, ne_energy
