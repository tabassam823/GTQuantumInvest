import numpy as np

def calculate_energy(x, h, J, N):
    Z = 1 - 2 * x 
    E = 0.0
    for i in range(N):
        E += h[i] * Z[i]
        for j in range(i + 1, N):
            E += J[i, j] * Z[i] * Z[j]
    return E
