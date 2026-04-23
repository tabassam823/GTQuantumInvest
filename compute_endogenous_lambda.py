import numpy as np

def compute_endogenous_lambda(mu_period, sigma_period):
    """
    Menghitung parameter risk-aversion (gamma) secara endogen berdasarkan
    Sharpe Ratio rata-rata lintas aset menggunakan variabel yang sudah 
    dihitung (termasuk volatility drag).
    
    Z = mu_avg / sigma_avg (Sharpe Ratio agregat)
    gamma = 1 / (1 + exp(Z))
    """
    mu_avg    = np.abs(mu_period).mean()
    sigma_avg = sigma_period.mean()
    
    if np.isnan(mu_avg) or np.isnan(sigma_avg) or sigma_avg == 0:
        return 0.5
        
    Z = mu_avg / sigma_avg   # Sharpe Ratio agregat
    gamma = 1.0 / (1.0 + np.exp(Z))
    
    return gamma
