import numpy as np

def calculate_expected_simple_return(simple_rets):
    """
    Menghitung expected simple return:
    mu_R = sum(R_t) / T
    """
    return simple_rets.mean()

def calculate_expected_log_return_with_drag(mu_simple, var_simple):
    """
    Menghitung expected log return dengan volatility drag:
    mu_r = mu_R - 0.5 * sigma_R^2
    """
    return mu_simple - 0.5 * var_simple
