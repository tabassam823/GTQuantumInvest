import numpy as np

def compute_endogenous_lambda(log_returns, tickers):
    """
    Menghitung parameter risk-aversion (gamma) secara endogen berdasarkan
    Sharpe Ratio rata-rata lintas aset, menggunakan skala periode 126 hari (~6 bulan).
    """
    mu_period    = log_returns[tickers].mean() * 126
    sigma_period = log_returns[tickers].std()  * np.sqrt(126)
    mu_avg    = abs(mu_period).mean()
    sigma_avg = sigma_period.mean()
    if np.isnan(mu_avg) or np.isnan(sigma_avg) or (mu_avg + sigma_avg) == 0:
        return 0.5
    Z = mu_avg / sigma_avg   # Sharpe Ratio agregat
    return 1.0 / (1.0 + np.exp(Z))
