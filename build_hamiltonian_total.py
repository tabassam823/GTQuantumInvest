import pennylane as qml

def build_hamiltonian_total(h_total, J_total, n_assets):
    """
    Membangun operator Hamiltonian Ising dari parameter h_i dan J_ij.
    """
    coeffs = []
    obs    = []

    for i in range(n_assets):
        if abs(h_total[i]) > 1e-10:
            coeffs.append(float(h_total[i]))
            obs.append(qml.PauliZ(i))

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if abs(J_total[i, j]) > 1e-10:
                coeffs.append(float(J_total[i, j]))
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    if len(coeffs) == 0:
        coeffs.append(0.0)
        obs.append(qml.Identity(0))

    return qml.Hamiltonian(coeffs, obs)
