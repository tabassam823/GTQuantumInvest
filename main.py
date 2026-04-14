import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from datetime import datetime

# Import fungsi-fungsi yang telah dipisahkan
from run_strategy_step import run_strategy_step
from rebalance_portfolio import rebalance_portfolio
from compute_metrics import compute_metrics
from compute_beta import compute_beta

warnings.filterwarnings('ignore')

# =============================================================================
# 0. Pembersihan File Lama
# =============================================================================
import os
files_to_remove = [
    'metrik_return_dan_lambda.csv', 'bias_h_total.csv', 'interaksi_J_total.csv',
    'riwayat_nash_sbr.csv', 'hasil_pencarian_lr.csv', 'hasil_depth_vs_energi.csv',
    'riwayat_iterasi_vqe.csv', 'theta_final_all_depths.csv'
]
for f in files_to_remove:
    if os.path.exists(f): os.remove(f)

# =============================================================================
# 1. Konfigurasi
# =============================================================================
tickers = ['BBCA.JK', 'TLKM.JK', 
           #'SMGR.JK', 'BMRI.JK',
           #'KLBF.JK', 'ASII.JK', 
           #'UNTR.JK', 'ICBP.JK',
           #'AMRT.JK', 'ADRO.JK',
           #'TPIA.JK', 'BBRI.JK'
           ]

N       = len(tickers)       # Jumlah aset kandidat
K       = 1                  # Jumlah aset target portofolio
penalty_A = 5.0              # Pengali Lagrange (A) untuk H_penalty (lambda)
max_depth = 6                # Kedalaman maksimum ansatz (adaptive)
maxiter   = 100              # Iterasi SPSA per depth-level

initial_capital = 100_000_000.0
lookback_days  = 126   # ~6 bulan perdagangan
rebalance_days = 21    # ~1 bulan perdagangan

benchmark_ticker = '^JKSE' # Indeks pembanding (bisa diubah, misal: '^JKSE' untuk IHSG, 'LQ45.JK' untuk LQ45, dll.)

# =============================================================================
# 2. Download Data
# =============================================================================
print("Mengunduh data saham...")
data = yf.download(tickers, start="2020-6-01", end="2024-01-01", progress=False)['Close']
data = data.dropna()
data_clean = data.sort_index()

print(f"Mengunduh data indeks pembanding ({benchmark_ticker})...")
benchmark_data = yf.download(benchmark_ticker, start='2021-01-04', end='2024-01-01', progress=False)['Close']
benchmark_data = benchmark_data.dropna()
benchmark_rets = benchmark_data.pct_change().dropna()

print(f"Data Berhasil Diunduh. Total hari observasi: {len(data_clean)}")

# --- EKSPOR DATA HARGA HARIAN (Poin 1) ---
data_clean.to_csv('harga_harian_saham.csv')
print("Data harga harian telah diekspor ke 'harga_harian_saham.csv'.")

# =============================================================================
# 3. Persiapan Backtest
# =============================================================================
start_bt_date    = pd.to_datetime('2021-01-04')
start_idx        = np.searchsorted(data_clean.index, start_bt_date)
rebalance_indices = range(start_idx, len(data_clean), rebalance_days)

value_vqe   = [initial_capital] * start_idx
value_bench = [initial_capital] * start_idx
value_assets = {t: [initial_capital] * start_idx for t in tickers}

holdings_vqe, holdings_bench = np.zeros(N), np.zeros(N)
cash_vqe, cash_bench = initial_capital, initial_capital
cash_assets    = {t: initial_capital for t in tickers}
holdings_assets = {t: 0.0 for t in tickers}

detail_logs = []

print(f"\n--- Memulai Backtest dari {data_clean.index[start_idx].date()} hingga {data_clean.index[-1].date()} ---")

# =============================================================================
# 4. Main Backtest Loop
# =============================================================================
last_lr_data = None

for i, curr_idx in enumerate(rebalance_indices):
    curr_date      = data_clean.index[curr_idx]
    train_start    = max(0, curr_idx - lookback_days)
    train_data     = data_clean.iloc[train_start:curr_idx]

    next_idx = (rebalance_indices[i + 1] if i + 1 < len(rebalance_indices) else len(data_clean))

    # Jalankan strategi step
    selected_indices, depth_used, energy_final, best_history, depth_energies, lr_data, ne_bs, ne_en = run_strategy_step(
        train_data, tickers, curr_date, K=K, penalty_A=penalty_A, max_depth=max_depth, maxiter=maxiter
    )
    
    selected_names = [tickers[idx] for idx in selected_indices]
    print(f"[{curr_date.date()}] VQE Terpilih: {selected_names} | Depth: {depth_used} | E_min: {energy_final:.6f}")

    # Logging detail
    vqe_details = "".join([f"    [Depth {d}] E = {en:.6f}\n" for d, en, iters in depth_energies])
    log_entry = (f"[{curr_date.date()}]\n  - Nash Eq: {ne_bs} | Energy: {ne_en:.6f}\n"
                 f"  - LR: {lr_data[2]:.4f}\n  - Detail:\n{vqe_details}"
                 f"  - Terpilih: {selected_names}\n" + "-"*40)
    detail_logs.append(log_entry)
    last_lr_data = lr_data

    # Rebalancing
    target_w_vqe = np.zeros(N)
    if len(selected_indices) > 0:
        w = 1.0 / len(selected_indices)
        for idx in selected_indices: target_w_vqe[idx] = w

    target_w_bench = np.full(N, 1.0 / N)
    current_prices = data_clean.iloc[curr_idx].values

    cash_vqe, holdings_vqe = rebalance_portfolio(cash_vqe, holdings_vqe, target_w_vqe, current_prices, N)

    if i == 0:
        cash_bench, holdings_bench = rebalance_portfolio(cash_bench, holdings_bench, target_w_bench, current_prices, N)
        for j, t in enumerate(tickers):
            target_w_indiv = np.zeros(N)
            target_w_indiv[j] = 1.0
            c_t, h_t = rebalance_portfolio(cash_assets[t], np.zeros(N), target_w_indiv, current_prices, N)
            cash_assets[t], holdings_assets[t] = c_t, h_t[j]

    # Simpan nilai harian hingga rebalance berikutnya
    for d in range(curr_idx, next_idx):
        prices = data_clean.iloc[d].values
        value_vqe.append(cash_vqe + np.sum(holdings_vqe * prices))
        value_bench.append(cash_bench + np.sum(holdings_bench * prices))
        for j, t in enumerate(tickers):
            value_assets[t].append(cash_assets[t] + holdings_assets[t] * prices[j])

print("\nBacktesting Selesai.")

# =============================================================================
# 5. Metrik & Beta
# =============================================================================
tr_vqe, sr_vqe, mdd_vqe = compute_metrics(value_vqe[start_idx-1:], initial_capital, "Quantum VQE")
tr_bench, sr_bench, mdd_bench = compute_metrics(value_bench[start_idx-1:], initial_capital, "Equal Weight")

bt_dates = data_clean.index[start_idx:]
vqe_rets = pd.Series(value_vqe[start_idx:], index=bt_dates).pct_change().dropna()
vqe_beta = compute_beta(vqe_rets, benchmark_rets.reindex(vqe_rets.index).fillna(0))
print(f"Beta VQE terhadap {benchmark_ticker}: {vqe_beta:.4f}")

# Simpan Laporan
with open("laporan_backtest.txt", "w") as f:
    f.write("LAPORAN STRATEGI QUANTUM VQE\n" + "="*30 + "\n")
    f.write(f"Return: {tr_vqe:.2f}%\nSharpe: {sr_vqe:.4f}\nBeta: {vqe_beta:.4f}\n\n")
    for log in detail_logs: f.write(log + "\n")

# =============================================================================
# 6. Visualisasi (Poin 6)
# =============================================================================
# 6a. Gambar Rangkaian Kuantum (Depth 6)
import pennylane as qml
dev_draw = qml.device("default.qubit", wires=N)
def circuit_to_draw(params):
    depth = 6
    w = params.reshape((depth + 1, N, 2))
    for layer in range(depth + 1):
        for q in range(N):
            qml.RY(w[layer, q, 0], wires=q)
            qml.RZ(w[layer, q, 1], wires=q)
        if layer < depth:
            for q in range(N - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[N - 1, 0])
    return qml.expval(qml.PauliZ(0))

qnode_draw = qml.QNode(circuit_to_draw, dev_draw)
params_dummy = np.zeros((7, N, 2))
fig_circ, ax_circ = qml.draw_mpl(qnode_draw)(params_dummy)
fig_circ.suptitle("Rangkaian Kuantum VQE (Depth 6)", fontsize=16)
plt.savefig('rangkaian_kuantum_depth6.png')
print("Gambar rangkaian kuantum disimpan sebagai 'rangkaian_kuantum_depth6.png'.")

# 6b. Grafik Tambahan (Hanya untuk Window Terakhir)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Konvergensi Energi (Iterasi SPSA)
spsa_iters = np.arange(1, len(best_history) + 1) * 25
axes[0].plot(spsa_iters, best_history, marker='o', markersize=4, color='b')
axes[0].set_title('Konvergensi Energi (Iterasi SPSA)')
axes[0].set_xlabel('Iterasi')
axes[0].set_ylabel('Energi')
axes[0].grid(True, linestyle='--', alpha=0.7)

# 2. Energi vs Depth
depths = [d for d, e, iters in depth_energies]
energies = [e for d, e, iters in depth_energies]
axes[1].plot(depths, energies, marker='s', markersize=8, color='r')
axes[1].set_title('Pencarian Energi vs Depth')
axes[1].set_xlabel('Depth')
axes[1].set_ylabel('Energi Minimum')
axes[1].set_xticks(depths)
axes[1].grid(True, linestyle='--', alpha=0.7)

# 3. LR Finder
test_a_vals, test_energies, chosen_a = last_lr_data
axes[2].plot(test_a_vals, test_energies, marker='o', color='g')
axes[2].axvline(chosen_a, color='red', linestyle='--', label=f'Best a: {chosen_a:.4f}')
axes[2].set_xscale('log')
axes[2].set_title('LR Finder (Window Terakhir)')
axes[2].set_xlabel('Learning Rate (a)')
axes[2].set_ylabel('Energy')
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].legend()

plt.tight_layout()
plt.savefig('grafik_konvergensi_detail.png')
print("Grafik konvergensi detail disimpan sebagai 'grafik_konvergensi_detail.png'.")

# 6c. Grafik Pertumbuhan Ekuitas (VQE vs Benchmark vs IHSG)
plt.figure(figsize=(12, 6))

# Hitung Pertumbuhan Indeks Pembanding untuk Grafik
benchmark_prices = benchmark_data.reindex(data_clean.index).ffill().bfill()
start_benchmark_price = benchmark_prices.loc[data_clean.index[start_idx]]
value_benchmark_idx = initial_capital * (benchmark_prices.iloc[start_idx:] / start_benchmark_price)

plt.plot(data_clean.index[:len(value_vqe)], value_vqe, label='Quantum VQE', linewidth=2.5, color='blue')
plt.plot(data_clean.index[:len(value_bench)], value_bench, label='Benchmark (Equal Weight)', linestyle='--', color='black', alpha=0.6)
plt.plot(value_benchmark_idx.index, value_benchmark_idx.values, label=f'Indeks ({benchmark_ticker})', linestyle='-.', color='magenta', linewidth=2)

plt.title(f'Perbandingan Pertumbuhan Ekuitas: VQE vs Equal Weight vs {benchmark_ticker}')
plt.ylabel('Total Ekuitas (Rupiah)')
plt.xlabel('Tanggal')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('hasil_backtest.png')
print("\nGrafik perbandingan lengkap disimpan sebagai 'hasil_backtest.png'.")
plt.show()
