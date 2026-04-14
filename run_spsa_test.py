import numpy as np

def run_spsa_test(cost_circuit, n_params, init_params=None, a_base=0.1, c_base=0.1, max_iters=30, batch_size=25):
    """Versi ringan SPSA untuk pengetesan Learning Rate."""
    rng = np.random.default_rng(42)
    if init_params is not None:
        params = init_params.copy()
    else:
        params = rng.uniform(0, 2 * np.pi, n_params)
    
    A_coeff = max_iters * 0.1
    alpha_exp = 0.602
    gamma_exp = 0.101
    
    total_iters = 0
    while total_iters < max_iters:
        for _ in range(batch_size):
            k = total_iters
            a_k = a_base / (A_coeff + k + 1) ** alpha_exp
            c_k = c_base / (k + 1) ** gamma_exp
            delta = 2 * rng.integers(0, 2, size=n_params) - 1
            cost_plus = float(cost_circuit(params + c_k * delta))
            cost_minus = float(cost_circuit(params - c_k * delta))
            grad = (cost_plus - cost_minus) / (2 * c_k * delta)
            params = params - a_k * grad
            total_iters += 1
    return float(cost_circuit(params))
