import pennylane as qml

def build_hamiltonian_total(h_total, J_total, n_assets, offset=0.0):
    """
    Membangun operator Hamiltonian Ising dari parameter h_i dan J_ij,
    termasuk konstanta pergeseran energi C_Ising.
    H = sum(h_i * Z_i) + sum(J_ij * Z_i * Z_j) + offset * I
    """
    coeffs = []
    obs    = []

    # Suku Bias (h_i * Z_i)
    for i in range(n_assets):
        if abs(h_total[i]) > 1e-12:
            coeffs.append(float(h_total[i]))
            obs.append(qml.PauliZ(i))

    # Suku Interaksi (J_ij * Z_i * Z_j)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if abs(J_total[i, j]) > 1e-12:
                coeffs.append(float(J_total[i, j]))
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    # Suku Konstanta (offset * Identity)
    if abs(offset) > 1e-12:
        coeffs.append(float(offset))
        obs.append(qml.Identity(0))

    if len(coeffs) == 0:
        coeffs.append(0.0)
        obs.append(qml.Identity(0))

    return qml.Hamiltonian(coeffs, obs)
