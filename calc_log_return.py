import numpy as np
import pandas as pd

def calculate_log_return(prices):
    """
    Menghitung log return secara manual berdasarkan rumus:
    r_t = ln(P_t / P_{t-1})
    """
    # Menggunakan shift(1) untuk mendapatkan P_{t-1}
    prev_prices = prices.shift(1)
    
    # Rumus: ln(P_t / P_{t-1})
    log_returns = np.log(prices / prev_prices)
    
    # Menghapus baris pertama (NaN) karena tidak memiliki P_{t-1}
    return log_returns.dropna()
