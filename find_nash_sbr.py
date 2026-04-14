import numpy as np
import pandas as pd
import os

def find_nash_sbr(h, J, curr_date, N=4, K=2, max_iters=100):
    """
    Pencarian Nash Equilibrium menggunakan Sequential Best Response (SBR).
    Mencatat riwayat iterasi dengan kolom tanggal.
    """
    current_selection = set(range(K))
    history = []
    
    def get_energy(selection):
        x = np.zeros(N)
        for idx in selection: x[idx] = 1
        Z = 1 - 2 * x
        E = np.dot(h, Z)
        for i in range(N):
            for j in range(i + 1, N):
                E += J[i, j] * Z[i] * Z[j]
        return E

    current_energy = get_energy(current_selection)
    history.append({
        'Date': curr_date.date(),
        'Iteration': 0,
        'Bitstring': "".join(str(int(i in current_selection)) for i in range(N)),
        'Energy': current_energy,
        'Swap': 'Initial'
    })
    
    for iteration in range(1, max_iters + 1):
        improved = False
        out_portfolio = set(range(N)) - current_selection
        
        best_swap = None
        min_energy = current_energy
        
        for i in current_selection:
            for j in out_portfolio:
                new_selection = (current_selection - {i}) | {j}
                new_energy = get_energy(new_selection)
                
                if new_energy < min_energy:
                    min_energy = new_energy
                    best_swap = (i, j)
        
        if best_swap:
            i, j = best_swap
            current_selection = (current_selection - {i}) | {j}
            current_energy = min_energy
            improved = True
            history.append({
                'Date': curr_date.date(),
                'Iteration': iteration,
                'Bitstring': "".join(str(int(idx in current_selection)) for idx in range(N)),
                'Energy': current_energy,
                'Swap': f"{i}<->{j}"
            })
        
        if not improved:
            break
            
    # --- EKSPOR RIWAYAT NASH SBR (Poin 3) ---
    df_history = pd.DataFrame(history)
    file_path = 'riwayat_nash_sbr.csv'
    if not os.path.exists(file_path):
        df_history.to_csv(file_path, index=False)
    else:
        df_history.to_csv(file_path, mode='a', header=False, index=False)

    final_x = np.zeros(N, dtype=int)
    for idx in current_selection: final_x[idx] = 1
    return "".join(str(bit) for bit in final_x), current_energy
