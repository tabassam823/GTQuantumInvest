import pandas as pd

def calculate_simple_return(prices):
    """
    Menghitung simple return secara manual berdasarkan rumus:
    R_t = (P_t - P_{t-1}) / P_{t-1}
    """
    # Menggunakan shift(1) untuk mendapatkan P_{t-1}
    prev_prices = prices.shift(1)
    
    # Rumus: (P_t - P_{t-1}) / P_{t-1}
    simple_returns = (prices - prev_prices) / prev_prices
    
    # Menghapus baris pertama (NaN) karena tidak memiliki P_{t-1}
    return simple_returns.dropna()
