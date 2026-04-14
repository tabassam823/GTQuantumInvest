import numpy as np

def calc_shannon_entropy(st_A):
    """[Tugas 1] Menghitung entropi Shannon H(X) dari distribusi probabilitas biner."""
    p1 = np.mean(st_A)
    p0 = 1.0 - p1
    H = 0.0
    if p0 > 0: H -= p0 * np.log2(p0)
    if p1 > 0: H -= p1 * np.log2(p1)
    return H
