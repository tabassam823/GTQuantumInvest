import numpy as np

def calc_classical_mutual_information(st_A, st_B):
    """[Tugas 1] Menghitung Classical Mutual Information I(X_i; X_j)."""
    n_ij = np.zeros((2, 2))
    for t in range(len(st_A)):
        n_ij[int(st_A[t]), int(st_B[t])] += 1
    
    prob_joint = n_ij / len(st_A)
    prob_A = prob_joint.sum(axis=1)
    prob_B = prob_joint.sum(axis=0)
    
    I_MI = 0.0
    for i in range(2):
        for j in range(2):
            if prob_joint[i, j] > 0:
                I_MI += prob_joint[i, j] * np.log2(prob_joint[i, j] / (prob_A[i] * prob_B[j]))
    return max(I_MI, 0.0)
