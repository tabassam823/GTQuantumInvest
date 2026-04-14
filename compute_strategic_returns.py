import numpy as np

def compute_strategic_returns(log_rets, binary_st, tickers):
    """
    [Tugas 2: Perhitungan Imbal Hasil Strategis]
    Menghitung imbal hasil strategis (mu_tilde_i) sebagai jumlahan terbobot
    dari ekspektasi bersyarat pada 16 microstates sistem 4 aset.
    """
    n_assets = len(tickers)
    total_days = len(log_rets)
    mu_tilde = np.zeros(n_assets)
    
    # Kelompokkan return berdasarkan binary states (microstates) dari semua aset
    grouped = log_rets.groupby([binary_st[t] for t in tickers])
    
    for state, group in grouped:
        P_s = len(group) / total_days           # Probabilitas microstate P(s)
        R_bar_s = group[tickers].mean().values  # Ekspektasi return bersyarat R_bar_i(s)
        mu_tilde += P_s * R_bar_s               # Jumlahan terbobot
        
    return mu_tilde
