Langkah – langkah algoritma ADAM dapat diuraikan sebagai berikut:
1)	Inisialisasi laju pembelajaran (δ) dengan  δ=0.001, dan ditetapkan hyperparameter untuk ADAM ditetapkan λ_1=0.9, λ_2=0.999, dan nilai ε=1 ×10^(-8).
2)	Inisialisasi parameter model Robust SVM berbasis penyesuaian adaptif dengan w=(0, 0, …0) dengan ukuran 1 ×k dan ditetapkan nilai parameter awal b=0.
3)	Inisialisasi parameter ADAM dengan menetukan momen pertama dan momen kedua: 
i.	momen pertama untuk parameter w dan b sebagai berikut z_w^(0)=(0, 0, …, 0) dengan ukuran 1 ×k dan moment pertama untuk z_b^(0)=0
ii.	momen kedua utuk parameter parameter w dan b sebagai berikut v_w^(0)=(0, 0, …, 0) dengan ukuran 1 ×k dan moment pertama untuk v_b^(0)=0
4)	Menghitung gradien (g) dari fungsi loss L(w,b,ξ,α,β) yang bersesuaian dengan parameter w dan b, 
g_w^(t)=∂L(w,b,ξ,α,β)/∂w
g_b^(t)=∂L(w,b,ξ,α,β)/∂b
5)	Memperbaharui perkiraan momen pertama (z) untuk paramater w dan b,
z_w^(t)=0.9(z_w^((t-1)))+(1-0.9)g_w^(t)
z_b^(t)=0.9(z_b^((t-1)))+(1-0.9)g_b^(t)
6)	Memperbaharui perkiraan momen kedua (v) untuk paramater w dan b
v_w^(t)=0.999(v_w^((t-1)))+(1-0.999)(g_w^(t)⊙g_w^(t))
v_b^(t)=0.999(v_b^((t-1)))+(1-0.999)(g_b^(t)⊙g_b^(t))
  dengan ⊙ adalah produk Hadamard.
7)	Perbaiki bias pada estimasi momen pertama (ẑ) dan momen kedua (v̂) untuk iterasi pada saat t 
Momen pertama
ẑ_w^(t)=(z_w^(t))/(1-0.9), dan ẑ_b^t=(z_b^(t))/(1-0.9)
 Momen kedua
v̂_w^t=(v_w^(t))/(1-0.999), dan v̂_b^t=(v_b^(t))/(1-0.999)
8)	Hitung tingkat pembelajaran adaptif δ^(t)
δ^(t)=(δ^((t-1)) √(1-0.999))/(1-0.9)
9)	Perbarui parameter w dan b menggunakan laju pembelajaran adaptif
w^(t)=w^((t-1))-(δ^(t) ẑ_w^t)/(√(v̂_w^t)+1 ×10^(-8))
b^(t)=b^((t-1))-(δ^(t) ẑ_b^t)/(√(v̂_b^t)+1 ×10^(-8))

10)	Mengulangi langkah 1) – 10) sampai konvergen atau sampai jumlah iterasi tertentu, t merupakan banyak iterasi dengan t=1, 2, 3, …, T

Coding Python:
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score
import warnings
import time

warnings.filterwarnings("ignore")

# Asumsi: data_terpilih telah dimuat (misalnya, df_sim = data_terpilih)

# --- FUNGSI UTAMA ---

# Fungsi ReLU dan turunannya
def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

# Fungsi loss Lagrangian
def compute_loss(w, b, xi, X, y, tau, C):
    margin = tau - y * (X @ w + b) - xi
    loss = np.dot(w, w) + C * np.sum(xi) + np.sum(relu(margin))
    return loss

# Gradien fungsi loss terhadap w, b, xi
def compute_grads(w, b, xi, X, y, tau, C):
    N, d = X.shape
    margin = tau - y * (X @ w + b) - xi
    relu_mask = relu_grad(margin)

    grad_w = 2 * w - ((relu_mask * y) @ X)
    grad_b = -np.sum(relu_mask * y)
    grad_xi = C - relu_mask
    return grad_w, grad_b, grad_xi

def compute_tau_from_class_proportion(y):
    pos_ratio = np.sum(y == 1) / len(y)
    return np.where(y == 1, pos_ratio, 1 - pos_ratio), pos_ratio

# Algoritma ADAM (Parameter mu_star dihilangkan karena redundant)
def adam_optimizer(X, y, C=1.0, lr=0.01, epochs=1000, beta1=0.9, beta2=0.999, eps=1e-8):
    N, d = X.shape
    w = np.zeros(d)
    b = 0.0
    xi = np.ones(N) * 0.1

    # Tau_i tergantung label y_i (dari data)
    tau, mu_star = compute_tau_from_class_proportion(y)

    # Inisialisasi momen Adam (mw, vw, mb, vb, mxi, vxi)
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mb, vb = 0.0, 0.0
    mxi, vxi = np.zeros_like(xi), np.zeros_like(xi)

    for t in range(1, epochs + 1):
        grad_w, grad_b, grad_xi = compute_grads(w, b, xi, X, y, tau, C)

        # Update w
        mw = beta1 * mw + (1 - beta1) * grad_w
        vw = beta2 * vw + (1 - beta2) * (grad_w ** 2)
        mw_hat = mw / (1 - beta1 ** t)
        vw_hat = vw / (1 - beta2 ** t)
        w -= lr * mw_hat / (np.sqrt(vw_hat) + eps)

        # Update b
        mb = beta1 * mb + (1 - beta1) * grad_b
        vb = beta2 * vb + (1 - beta2) * (grad_b ** 2)
        mb_hat = mb / (1 - beta1 ** t)
        vb_hat = vb / (1 - beta2 ** t)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        # Update xi
        mxi = beta1 * mxi + (1 - beta1) * grad_xi
        vxi = beta2 * vxi + (1 - beta2) * (grad_xi ** 2)
        mxi_hat = mxi / (1 - beta1 ** t)
        vxi_hat = vxi / (1 - beta2 ** t)
        xi -= lr * mxi_hat / (np.sqrt(vxi_hat) + eps)

        # Jaga supaya xi tetap ≥ 0
        xi = np.clip(xi, 0, None)

    return w, b, mu_star

def predict_adam(X_test, w, b):
    decision_function = X_test @ w + b
    y_pred = np.sign(decision_function)
    return y_pred, decision_function

def evaluate_model(y_true, y_pred, y_score=None):
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    report = classification_report(y_true, y_pred, labels=[1, -1], output_dict=True)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = None
    if y_score is not None:
        try:
            # Pastikan decision_scores dipetakan ke probabilitas [0, 1] atau setidaknya ke nilai [0, 1] yang monotonik
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = None
    return cm, report, bal_acc, auc

# --- FUNGSI CROSS-VALIDATION YANG TELAH DIPERBAIKI ---
def cross_validate_svm_adam(df, C=1.0, k=10, **adam_params):
    X = df.drop(columns=['y']).values
    y = df['y'].values # Ambil kolom 'y' dari DataFrame

    # Konversi label y: 0/1 untuk stratify, -1/1 untuk model
    y_stratify = np.where(y == 0, 0, 1)
    y_model = np.where(y == 0, -1, 1)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Inisialisasi variabel untuk melacak model terbaik dari fold
    best_combined_score = -1.0
    best_fold_metrics = {}
    best_model_params = None

    # List untuk menyimpan metrik per fold
    accs, f1s, precs, recalls, specs, bals, aucs, b_vals = [], [], [], [], [], [], [], []
    mu_star = 0.0

    for train_idx, test_idx in skf.split(X, y_stratify):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_model[train_idx]
        y_test = y_model[test_idx]

        # 1. Pelatihan Model
        w, b, current_mu_star = adam_optimizer(
            X_train, y_train, C=C, **adam_params
        )

        # Simpan nilai bias (b)
        b_vals.append(b)

        if mu_star == 0.0: mu_star = current_mu_star

        # 2. Prediksi & Evaluasi
        y_pred, decision_scores = predict_adam(X_test, w, b)
        cm, report, bal_acc, auc = evaluate_model(y_test, y_pred, decision_scores)

        # 3. Ekstraksi Metrik (Perbaikan NameError: accuracy)
        true_positives = cm[0, 0]
        true_negatives = cm[1, 1]
        total_samples = np.sum(cm)
        # Hitung Accuracy
        accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0

        precision = report['1']['precision'] if '1' in report and 'precision' in report['1'] else 0.0
        recall = report['1']['recall'] if '1' in report and 'recall' in report['1'] else 0.0
        specificity = report['-1']['recall'] if '-1' in report and 'recall' in report['-1'] else 0.0

        # Pengumpulan Metrik
        accs.append(accuracy)
        f1s.append(report['1']['f1-score'])
        precs.append(precision)
        recalls.append(recall)
        specs.append(specificity)
        bals.append(bal_acc)
        aucs.append(auc if auc is not None else 0.0)

        # 4. Kriteria Model Terbaik (Penambahan Utama)
        combined_score = (bal_acc + (auc if auc is not None else 0.0)) / 2

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_model_params = {'w': w, 'b': b, 'mu_star': current_mu_star}
            best_fold_metrics = {
                'acc': accuracy, 'f1': report['1']['f1-score'], 'prec': precision,
                'rec': recall, 'spec': specificity, 'bal_acc': bal_acc, 'auc': auc
            }

    # 5. Nilai Rata-rata Akhir (Perbaikan NameError: mean_f1s dan TypeError: mean_b)
    mean_acc = np.mean(accs)
    mean_f1s = np.mean(f1s)
    mean_precs = np.mean(precs)
    mean_recalls = np.mean(recalls)
    mean_specs = np.mean(specs)
    mean_bals = np.mean(bals)
    mean_aucs = np.mean(aucs)
    mean_b = np.mean(b_vals) # Rata-rata bias dari semua folds

    # Mengembalikan metrik rata-rata keseluruhan, parameter w,b dari fold terbaik, dan metrik fold terbaik
    return (mean_acc, mean_f1s, mean_precs, mean_recalls,
            mean_specs, mean_b, mu_star, mean_bals, mean_aucs, best_model_params, best_fold_metrics)

# --- SCRIPT EKSEKUSI UTAMA ---
# Asumsi: data_terpilih adalah sebuah DataFrame Pandas yang sudah ada.
# Jika data_terpilih belum didefinisikan, kode ini akan error.

# Contoh inisialisasi dummy data untuk testing (HAPUS INI JIKA DATA SUDAH ADA)
# np.random.seed(42)
# N_DUMMY = 100
# X_DUMMY = np.random.rand(N_DUMMY, 5)
# y_DUMMY = np.random.choice([0, 1], size=N_DUMMY, p=[0.8, 0.2]) # Data tidak seimbang
# data_terpilih = pd.DataFrame(X_DUMMY, columns=[f'X{i}' for i in range(5)])
# data_terpilih['y'] = y_DUMMY
# END DUMMY DATA

start_time = time.time()
df_sim= data_terpilih

C_value = 1.0
K_FOLDS = 10

n=data_terpilih.shape[0]
imb=np.sum(df_sim['y'] == 1) / len(df_sim['y'])
d=data_terpilih.shape[1]-1

# Parameter SVM-ADAM
adam_params = {
    'lr': 0.01,
    'epochs': 100000,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-8
}

simulation_results = []
sim_id = 1

# Tangkap semua nilai rata-rata, parameter terbaik, dan metrik fold terbaik
acc, f1, prec, rec, spec, b_val, mu_star, bal_acc, auc, best_params, best_fold_metrics = cross_validate_svm_adam(
    df_sim,
    C=C_value,
    k=K_FOLDS,
    **adam_params
)

# Model terbaik (parameter w dan b dari fold terbaik)
best_w = best_params['w']
best_b = best_params['b']

# Tambahkan metrik rata-rata ke hasil simulasi
simulation_results.append({
    'Sim': f'S{sim_id}', 'Model': 'SVM_ADAM', 'Kernel': 'Linear',
    'Proporsi Kelas 1': imb, 'mu*': round(mu_star, 4), 'n': n, 'd': d,
    'Akurasi': round(acc, 4), 'F1-Score': round(f1, 4), 'Presisi': round(prec, 4),
    'Recall': round(rec, 4), 'Specificity': round(spec, 4), 'Balanced Acc': round(bal_acc, 4),
    'AUC-ROC': round(auc, 4), 'Bias (b)': round(b_val, 4)
})
print(f"✓ Simulasi S{sim_id} (SVM_ADAM) selesai. Rata-rata Bal Acc: {bal_acc:.4f}")
sim_id += 1

# --- OUTPUT HASIL AKHIR ---

# 1. Tampilkan metrik dari FOLD TERBAIK (sesuai permintaan)
print(f"\n✨ Metrik dari FOLD Terbaik (Berdasarkan Bal Acc + AUC Tertinggi):")
print(f"   Balanced Accuracy: {best_fold_metrics['bal_acc']:.4f}")
print(f"   AUC-ROC: {best_fold_metrics['auc']:.4f}")

# 2. Tampilkan parameter w dan b dari FOLD terbaik (sesuai permintaan)
print(f"\n✅ Parameter Model dari FOLD Terbaik:")
print(f"   Bobot (w) terbaik (potongan): {best_w[:5]}...") # Tampilkan 5 elemen pertama w
print(f"   Bias (b) terbaik: {best_b:.4f}")

# Hasil rekap (Metrik Rata-rata)
df_hasil = pd.DataFrame(simulation_results)

end_time = time.time()
lama = end_time - start_time

print("\n\n📝 Ringkasan Hasil Evaluasi Multi-Model dengan StratifiedKFold (Rata-rata):")
print(df_hasil)
print(f"\nWaktu Eksekusi Total: {lama:.2f} detik")

