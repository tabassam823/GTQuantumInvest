import numpy as np
import pandas as pd
import os
from compute_endogenous_lambda import compute_endogenous_lambda
from find_nash_sbr import find_nash_sbr
from build_hamiltonian_total import build_hamiltonian_total
from find_optimal_lr_spsa import find_optimal_lr_spsa
from run_vqe_adaptive import run_vqe_adaptive
from calc_simple_return import calculate_simple_return
from calc_log_return import calculate_log_return
from calc_expected_returns import calculate_expected_simple_return, calculate_expected_log_return_with_drag
from calc_expected_returns import calculate_expected_simple_return, calculate_expected_log_return_with_drag

def run_strategy_step(lookback_data, tickers, curr_date, K=2, penalty_A=5.0,
                      max_depth=6, maxiter=100,
                      max_total_iter=2000, batch_size=25,
                      conv_window=4, conv_tol=1e-4):
    """
    Eksekusi satu periode pembelajaran dengan pencatatan tanggal.
    """
    n_assets = len(tickers)
    # Simple Return dihitung manual via fungsi baru
    simple_rets = calculate_simple_return(lookback_data)
    # Log Return dihitung manual via fungsi baru
    log_rets  = calculate_log_return(lookback_data)
    
    binary_st = (log_rets <= 0).astype(int)   # 1 = turun, 0 = naik

    # --- PERHITUNGAN EXPECTED RETURN (Rumus Volatility Drag) ---
    mu_R_daily = calculate_expected_simple_return(simple_rets)
    # Variance log return sebagai dasar volatility drag
    var_r_daily = log_rets.var() 
    mu_r_daily = calculate_expected_log_return_with_drag(mu_R_daily, var_r_daily)

    # Menggunakan return periode (126 hari)
    mu_simple_period = mu_R_daily.values * 126
    mu_log_period = mu_r_daily.values * 126
    
    sigma_log = log_rets.std().values
    sigma_period_log = sigma_log * np.sqrt(126)
    
    # Matriks kovariansi tetap berbasis log return
    sigma_period_matrix = log_rets.cov().values * 126

    # gamma disini mewakili degree of risk-aversion, menggunakan mu_log_period (dengan drag) dan sigma_period_log
    gamma = compute_endogenous_lambda(mu_log_period, sigma_period_log)
    lam = penalty_A # penalty lambda untuk pembatas kardinalitas
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

    # --- KONSTRUKSI MATRIKS Q (QUBO) ---
    Q_diag = np.zeros(n_assets)
    Q_off  = np.zeros((n_assets, n_assets))
    K_sq = K**2
    
    for i in range(n_assets):
        # Q_ii = (gamma * sigma_i^2) / (2 * K^2) - (mu_i / K) + lam * (1 - 2*K)
        Q_diag[i] = (gamma * sigma_period_matrix[i, i]) / (2.0 * K_sq) - (mu_simple_period[i] / K) + lam * (1.0 - 2.0 * K)
        for j in range(i + 1, n_assets):
            # Q_ij = (gamma * sigma_ij) / (2 * K^2) + lam
            Q_val = (gamma * sigma_period_matrix[i, j]) / (2.0 * K_sq) + lam
            Q_off[i, j] = Q_val
            Q_off[j, i] = Q_val

    # --- KONSTRUKSI PARAMETER ISING (h, J, C_Ising) ---
    h_total = np.zeros(n_assets)
    J_total = np.zeros((n_assets, n_assets))
    C_const = lam * K_sq
    
    for i in range(n_assets):
        sum_Q_ij_half = 0.0
        for j in range(n_assets):
            if i != j:
                sum_Q_ij_half += Q_off[i, j] / 2.0
                if i < j:
                    # J_ij = Q_ij / 4
                    J_total[i, j] = Q_off[i, j] / 4.0
                    J_total[j, i] = Q_off[i, j] / 4.0
        
        # h_i = Q_ii / 2 + sum_{j!=i} (Q_ij / 2)
        h_total[i] = (Q_diag[i] / 2.0) + sum_Q_ij_half

    # C_Ising = sum(Q_ii / 2) + sum_{i!=j} (Q_ij / 4) + C
    sum_Q_ii_half = np.sum(Q_diag) / 2.0
    sum_Q_ij_quarter = 0.0
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            sum_Q_ij_quarter += Q_off[i, j] / 2.0 # sum_{i!=j} Q_ij/4 = sum_{i<j} Q_ij/2
            
    C_Ising = sum_Q_ii_half + sum_Q_ij_quarter + C_const

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

    H = build_hamiltonian_total(h_total, J_total, n_assets, offset=C_Ising)
    
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
