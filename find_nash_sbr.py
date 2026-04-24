import numpy as np
import pandas as pd
import os

def find_nash_sbr(mu, sigma_matrix, gamma, curr_date, N=4, K=2, max_iters=100):
    """
    Pencarian Nash Equilibrium menggunakan Sequential Best Response (SBR)
    berdasarkan Utilitas Finansial:
    u_i = x_i * (mu_i - (gamma/2)*sigma_ii - gamma * sum_{j!=i} sigma_ij * x_j)
    """
    # Inisialisasi: pilih K aset pertama
    current_selection = set(range(K))
    history = []
    
    def calculate_total_utility(selection):
        x = np.zeros(N)
        for idx in selection: x[idx] = 1
        
        total_u = 0.0
        for i in selection:
            # sum_{j!=i} sigma_ij * x_j
            interaction = 0.0
            for j in range(N):
                if i != j:
                    interaction += sigma_matrix[i, j] * x[j]
            
            u_i = mu[i] - (gamma / 2.0) * sigma_matrix[i, i] - gamma * interaction
            total_u += u_i
        return total_u

    current_u = calculate_total_utility(current_selection)
    
    history.append({
        'Date': curr_date.date(),
        'Iteration': 0,
        'Bitstring': "".join(str(int(i in current_selection)) for i in range(N)),
        'Utility': current_u,
        'Swap': 'Initial'
    })
    
    for iteration in range(1, max_iters + 1):
        improved = False
        out_portfolio = set(range(N)) - current_selection
        
        best_swap = None
        max_utility = current_u
        
        # SBR: Coba tukar satu aset di dalam dengan satu aset di luar
        for i in current_selection:
            for j in out_portfolio:
                new_selection = (current_selection - {i}) | {j}
                new_u = calculate_total_utility(new_selection)
                
                if new_u > max_utility:
                    max_utility = new_u
                    best_swap = (i, j)
        
        if best_swap:
            i, j = best_swap
            current_selection = (current_selection - {i}) | {j}
            current_u = max_utility
            improved = True
            history.append({
                'Date': curr_date.date(),
                'Iteration': iteration,
                'Bitstring': "".join(str(int(idx in current_selection)) for idx in range(N)),
                'Utility': current_u,
                'Swap': f"{i}<->{j}"
            })
        
        if not improved:
            break
            
    # --- EKSPOR RIWAYAT NASH SBR ---
    df_history = pd.DataFrame(history)
    file_path = 'riwayat_nash_sbr.csv'
    if not os.path.exists(file_path):
        df_history.to_csv(file_path, index=False)
    else:
        df_history.to_csv(file_path, mode='a', header=False, index=False)

    final_x = np.zeros(N, dtype=int)
    for idx in current_selection: final_x[idx] = 1
    # Kembalikan bitstring dan utilitas akhir (sebagai pengganti energi)
    return "".join(str(bit) for bit in final_x), current_u
