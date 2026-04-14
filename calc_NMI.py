import numpy as np
from calc_classical_mutual_information import calc_classical_mutual_information
from calc_shannon_entropy import calc_shannon_entropy

def calc_NMI(st_A, st_B):
    """
    [Tugas 1] Menghitung Normalized Mutual Information (NMI) 
    menggunakan Upper Bound Theorem: NMI(i,j) = I(X_i:X_j) / sqrt(H(X_i)*H(X_j)).
    """
    I_AB = calc_classical_mutual_information(st_A, st_B)
    H_A = calc_shannon_entropy(st_A)
    H_B = calc_shannon_entropy(st_B)
    
    if H_A == 0 or H_B == 0:
        return 0.0
    
    return I_AB / np.sqrt(H_A * H_B)
