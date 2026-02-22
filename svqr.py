"""
SVQR-LASSO pada Data Return Saham BBCA
=======================================
Kode ini menerapkan Support Vector Quantile Regression (SVQR)
dengan dan tanpa regularisasi LASSO untuk memprediksi return saham BBCA.

Semua logika ditulis langsung (tanpa function/class) agar mudah dibaca
dan dipelajari baris per baris.
"""

# ─────────────────────────────────────────────
# IMPORT LIBRARY
# ─────────────────────────────────────────────
import numpy as np                      # untuk operasi matematika array
import pandas as pd                     # untuk membaca file Excel
from scipy.optimize import minimize     # untuk optimasi numerik (SLSQP)


# ─────────────────────────────────────────────
# BAGIAN 1 — LOAD DATA
# ─────────────────────────────────────────────

# Baca file Excel yang berisi return saham (target y)
data_y = pd.read_excel("data_empiris.xlsx")

# Baca file Excel yang berisi semua fitur/prediktor (X)
data_X = pd.read_excel("var_empiris.xlsx")

# Nama kolom target yang ingin diprediksi
nama_target = "BBCA"

# Ambil semua nama kolom fitur KECUALI BBCA
nama_fitur = [kolom for kolom in data_X.columns if kolom != nama_target]

# Ambil 422 data terakhir sebagai array numpy untuk target (y)
y = data_y[nama_target].tail(422).values

# Ambil semua baris data fitur sebagai array numpy
X = data_X[nama_fitur].values

# Jumlah observasi (baris) dan jumlah fitur (kolom)
n_obs, n_fitur = X.shape   # n_obs = banyak data, n_fitur = banyak variabel input

print("=" * 65)
print(f"  TARGET  : {nama_target}")
print(f"  Fitur   : {n_fitur} kolom")
print(f"  Observasi: {n_obs} baris")
print(f"  y — min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.5f}")
print()


# ─────────────────────────────────────────────
# BAGIAN 2 — KONFIGURASI MODEL
# ─────────────────────────────────────────────

# Daftar kuantil yang akan diuji
# tau=0.01 → kuantil sangat bawah (ekor kiri ekstrem)
# tau=0.05 → kuantil bawah
# tau=0.1  → kuantil bawah moderat
daftar_tau = [0.01, 0.05, 0.1]

# Daftar nilai lambda yang akan diuji
# lambda=0.0        → SVQR biasa (tanpa LASSO)
# lambda=0.0004...  → SVQR-LASSO (dengan regularisasi L1)
daftar_lambda = [0.0, 0.0004883464]

# Parameter tetap model
gamma   = 1.0    # penalti pelanggaran batas kuantil (semakin besar = lebih ketat)
epsilon = 0.01   # lebar "tube" — batas toleransi kesalahan prediksi


# ─────────────────────────────────────────────
# BAGIAN 3 — LOOP UTAMA: TRAINING & EVALUASI
# ─────────────────────────────────────────────

# Daftar untuk menyimpan ringkasan hasil semua model
semua_hasil = []

for tau in daftar_tau:
    # tau adalah kuantil yang sedang diproses (0.01, 0.05, atau 0.1)

    print("=" * 65)
    print(f"  KUANTIL tau = {tau}")
    print("=" * 65)

    for lam in daftar_lambda:
        # lam adalah nilai lambda yang sedang diproses (0 atau 0.000488...)

        pakai_lasso = lam > 0   # True jika ini adalah model SVQR-LASSO

        # ─── Definisi indeks variabel optimasi ───────────────────────
        # Semua variabel disusun dalam satu vektor panjang:
        # [ w(0..n_fitur-1) | b(n_fitur) | xi(n_fitur+1..n_fitur+n_obs) | xi*(..2*n_obs) | v(..) ]

        idx_w      = slice(0, n_fitur)                                          # indeks bobot fitur
        idx_b      = n_fitur                                                     # indeks intercept
        idx_xi     = slice(n_fitur + 1, n_fitur + 1 + n_obs)                    # indeks slack atas (xi)
        idx_xis    = slice(n_fitur + 1 + n_obs, n_fitur + 1 + 2*n_obs)          # indeks slack bawah (xi*)
        idx_v      = slice(n_fitur + 1 + 2*n_obs, n_fitur + 1 + 2*n_obs + n_fitur)  # indeks variabel bantu LASSO

        # Total jumlah variabel dalam vektor optimasi
        if pakai_lasso:
            n_var = n_fitur + 1 + 2*n_obs + n_fitur   # tambah n_fitur variabel v untuk LASSO
        else:
            n_var = n_fitur + 1 + 2*n_obs               # tanpa variabel v

        # ─── Definisi fungsi objektif ─────────────────────────────────
        def fungsi_objektif(params):
            """
            Fungsi yang diminimumkan oleh optimizer.

            Tanpa LASSO (persamaan 2.43):
              0.5 * ||w||^2  +  gamma * sum(tau*xi + (1-tau)*xi*)

            Dengan LASSO (persamaan 2.43 + L1):
              0.5 * ||w||^2  +  lambda * sum(v)  +  gamma * sum(tau*xi + (1-tau)*xi*)
            """
            w   = params[idx_w]     # vektor bobot fitur
            xi  = params[idx_xi]    # slack variable atas
            xis = params[idx_xis]   # slack variable bawah

            # Suku regularisasi ridge: 0.5 * ||w||^2
            ridge = 0.5 * np.dot(w, w)

            # Suku penalti slack: gamma * sum(tau*xi + (1-tau)*xi*)
            slack = gamma * np.sum(tau * xi + (1 - tau) * xis)

            if pakai_lasso:
                v = params[idx_v]           # variabel bantu linearisasi |w|
                lasso = lam * np.sum(v)     # suku LASSO: lambda * sum(v)
                return ridge + lasso + slack
            else:
                return ridge + slack

        # ─── Definisi kendala (constraints) ───────────────────────────
        # Optimizer SLSQP mensyaratkan semua kendala dalam bentuk >= 0

        kendala = []

        for t in range(n_obs):
            # xt = vektor fitur untuk observasi ke-t
            # yt = nilai target untuk observasi ke-t
            xt = X[t]
            yt = y[t]

            # Kendala 1: epsilon + xi_t - (yt - w^T*xt - b) >= 0
            # → prediksi tidak boleh terlalu jauh di atas nilai aktual
            def k1(params, xt=xt, yt=yt):
                prediksi = params[idx_w] @ xt + params[idx_b]   # w^T*x + b
                return epsilon + params[idx_xi.start + (t - 0)] - (yt - prediksi)

            # Kendala 2: epsilon + xi*_t - (w^T*xt + b - yt) >= 0
            # → prediksi tidak boleh terlalu jauh di bawah nilai aktual
            def k2(params, xt=xt, yt=yt):
                prediksi = params[idx_w] @ xt + params[idx_b]
                return epsilon + params[idx_xis.start + (t - 0)] - (prediksi - yt)

            # Kendala 3: xi_t >= 0 (slack atas tidak boleh negatif)
            def k3(params, t=t):
                return params[n_fitur + 1 + t]

            # Kendala 4: xi*_t >= 0 (slack bawah tidak boleh negatif)
            def k4(params, t=t):
                return params[n_fitur + 1 + n_obs + t]

            kendala += [
                {'type': 'ineq', 'fun': k1},
                {'type': 'ineq', 'fun': k2},
                {'type': 'ineq', 'fun': k3},
                {'type': 'ineq', 'fun': k4},
            ]

        # Kendala tambahan untuk LASSO: v_j >= |w_j|
        # Ini dilakukan dengan dua kendala linear:
        #   v_j - w_j >= 0  dan  v_j + w_j >= 0
        if pakai_lasso:
            for j in range(n_fitur):
                def k5(params, j=j):
                    return params[n_fitur + 1 + 2*n_obs + j] - params[j]  # v_j >= w_j

                def k6(params, j=j):
                    return params[n_fitur + 1 + 2*n_obs + j] + params[j]  # v_j >= -w_j

                kendala += [
                    {'type': 'ineq', 'fun': k5},
                    {'type': 'ineq', 'fun': k6},
                ]

        # ─── Inisialisasi & Optimasi ───────────────────────────────────
        # Titik awal semua variabel = 0 (vektor nol)
        titik_awal = np.zeros(n_var)

        # Jalankan optimasi SLSQP (Sequential Least Squares Programming)
        hasil_optim = minimize(
            fungsi_objektif,   # fungsi yang diminimumkan
            titik_awal,        # tebakan awal
            method='SLSQP',    # metode optimasi (cocok untuk kendala ineq)
            constraints=kendala,
            options={
                'ftol': 1e-9,       # toleransi konvergensi (presisi tinggi)
                'maxiter': 2000,    # maksimum iterasi
                'disp': False       # tidak tampilkan log iterasi
            }
        )

        # ─── Ekstrak hasil ──────────────────────────────────────────────
        bobot     = hasil_optim.x[idx_w]    # w: koefisien tiap fitur
        intercept = hasil_optim.x[idx_b]    # b: nilai intercept/bias
        nilai_loss = hasil_optim.fun         # nilai fungsi objektif akhir

        # Hitung prediksi: q_tau(x) = w^T * x + b  (persamaan 2.41)
        prediksi = X @ bobot + intercept

        # Hitung Pinball Loss (Check Function rata-rata) pada data training
        # Ini adalah metrik evaluasi utama untuk quantile regression
        residu = y - prediksi
        # Check function: tau*|u| jika u>=0, (1-tau)*|u| jika u<0
        check  = np.where(residu >= 0, tau * np.abs(residu), (1 - tau) * np.abs(residu))
        pinball = np.mean(check)   # rata-rata pinball loss

        # Hitung jumlah fitur aktif (|w_j| > 0.0001 dianggap aktif)
        n_aktif = int(np.sum(np.abs(bobot) > 1e-4))

        # ─── Tampilkan Hasil ────────────────────────────────────────────
        nama_model = "SVQR-LASSO" if pakai_lasso else "SVQR      "
        print("=" * 55)
        print(f"  {nama_model}  tau={tau}, lambda={lam}")
        print("=" * 55)

        # Tampilkan bobot beserta nama fitur masing-masing
        # Format: [indeks] nama_fitur : nilai_bobot
        print(f"  Bobot w (per fitur):")
        for i, (nama, w_val) in enumerate(zip(nama_fitur, np.round(bobot, 4))):
            print(f"    [{i}] {nama:25s} : {w_val:.4f}")

        print(f"  Intercept: {intercept:.4f}")

        if pakai_lasso:
            print(f"  ||w||_1  : {np.sum(np.abs(bobot)):.4f}  (norma L1 / LASSO)")
            print(f"  Fitur aktif: {n_aktif} dari {n_fitur}")

        print(f"  Loss Objektif : {nilai_loss:.6f}")
        print(f"  Pinball Loss  : {pinball:.6f}")
        print()

        # Simpan ringkasan ke daftar (tanpa info top-5 fitur)
        semua_hasil.append({
            "tau"     : tau,
            "model"   : "SVQR-LASSO" if pakai_lasso else "SVQR",
            "lambda"  : lam,
            "pinball" : round(pinball, 6),
            "n_aktif" : n_aktif,
        })


# ─────────────────────────────────────────────
# BAGIAN 4 — RINGKASAN AKHIR
# ─────────────────────────────────────────────

print("=" * 65)
print("  RINGKASAN SEMUA MODEL")
print("=" * 65)

# Ubah daftar hasil menjadi DataFrame agar mudah dibaca
df_ringkasan = pd.DataFrame(semua_hasil)

# Tampilkan tabel ringkasan
print(df_ringkasan.to_string(index=False))