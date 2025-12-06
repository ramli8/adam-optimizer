# ### LOAD LIBRARY

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# ### DATA LOADING FROM EXCEL

# Membaca data dari file data.xlsx
# Struktur file:
# Baris 1: Header (akan di-skip)
# Baris 2 dst: Data
# Kolom 1: y (response variable)
# Kolom 2: x1 (predictor 1)
# Kolom 3: x2 (predictor 2)
# Kolom 4: x3 (predictor 3)
# Kolom 5: x4 (predictor 4)

print("Loading data from data.xlsx...")
data = pd.read_excel('data.xlsx', header=0)  # Header di baris 1

# Ekstrak kolom sesuai struktur
y = data.iloc[:, 0].values  # Kolom 1 (index 0) = y
x1 = data.iloc[:, 1].values  # Kolom 2 (index 1) = x1
x2 = data.iloc[:, 2].values  # Kolom 3 (index 2) = x2
x3 = data.iloc[:, 3].values  # Kolom 4 (index 3) = x3
x4 = data.iloc[:, 4].values  # Kolom 5 (index 4) = x4

print(f"Data loaded successfully!")
print(f"Number of samples: {len(y)}")
print(f"y shape: {y.shape}")

# MENYESUAIKAN SHAPE
# X memiliki 4 kolom (x1, x2, x3, x4)
X = np.column_stack([x1, x2, x3, x4])  # Shape: (n_samples, 4)
print(f"X shape: {X.shape}")  # (n_samples, 4)
print(f"\nFirst 5 rows of data:")
print(f"y: {y[:5]}")
print(f"x1: {x1[:5]}")
print(f"x2: {x2[:5]}")
print(f"x3: {x3[:5]}")
print(f"x4: {x4[:5]}")

# ### CLASS SVQR

class AccurateSVQR(BaseEstimator, RegressorMixin):
    """
    Accurate Support Vector Quantile Regression with Pinball Loss
    
    This implementation uses the correct pinball loss function and 
    a custom solver based on quadratic programming formulation.
    """
    
    def __init__(self, tau=0.5, kernel='rbf', C=1.0, gamma='auto', 
                 epsilon=1e-6, max_iter=1000, tol=1e-6):
        """
        Parameters:
        -----------
        tau : float, default=0.5
            Quantile level (0 < tau < 1)
        kernel : str, default='rbf'
            Kernel type ('linear', 'rbf', 'poly')
        C : float, default=1.0
            Regularization parameter
        gamma : float or 'auto', default='auto'
            Kernel coefficient for RBF kernel
        epsilon : float, default=1e-6
            Precision tolerance
        max_iter : int, default=1000
            Maximum iterations for solver
        tol : float, default=1e-6
            Convergence tolerance
        """
        self.tau = tau
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        
    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix"""
        if X2 is None:
            X2 = X1
            
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            if self.gamma == 'auto':
                gamma = 1.0 / X1.shape[1]
            else:
                gamma = self.gamma
                
            # Efficient RBF kernel computation
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * np.maximum(distances, 0))
        elif self.kernel == 'poly':
            degree = getattr(self, 'degree', 3)
            return (np.dot(X1, X2.T) + 1) ** degree
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _pinball_loss(self, y_true, y_pred):
        """Pinball loss function (quantile loss)"""
        error = y_true - y_pred
        return np.mean(np.maximum(self.tau * error, (self.tau - 1) * error))
    
    def _pinball_loss_derivative(self, error):
        """Derivative of pinball loss"""
        return np.where(error >= 0, -self.tau, self.tau - 1)
    
    def _solve_dual_problem(self, K, y):
        """
        Solve the dual problem for SVQR using quadratic programming approach
        
        The dual problem for SVQR is:
        maximize: sum(y_i * (alpha_i^+ - alpha_i^-)) - 0.5 * sum_ij (alpha_i^+ - alpha_i^-)(alpha_j^+ - alpha_j^-) K_ij
        subject to: 0 <= alpha_i^+ <= C*tau, 0 <= alpha_i^- <= C*(1-tau)
                   sum(alpha_i^+ - alpha_i^-) = 0
        """
        n = len(y)
        
        # Variables: [alpha_plus, alpha_minus]
        # Each of size n, so total 2n variables
        
        def objective(alpha):
            alpha_plus = alpha[:n]
            alpha_minus = alpha[n:]
            alpha_diff = alpha_plus - alpha_minus
            
            # Objective: -[sum(y_i * alpha_diff_i) - 0.5 * alpha_diff^T K alpha_diff]
            linear_term = np.dot(y, alpha_diff)
            quadratic_term = 0.5 * np.dot(alpha_diff, np.dot(K, alpha_diff))
            return -(linear_term - quadratic_term)
        
        def objective_grad(alpha):
            alpha_plus = alpha[:n]
            alpha_minus = alpha[n:]
            alpha_diff = alpha_plus - alpha_minus
            
            grad_diff = -y + np.dot(K, alpha_diff)
            grad = np.zeros(2*n)
            grad[:n] = grad_diff      # gradient w.r.t alpha_plus
            grad[n:] = -grad_diff     # gradient w.r.t alpha_minus
            return grad
        
        # Constraints
        constraints = []
        
        # Sum constraint: sum(alpha_plus - alpha_minus) = 0
        A_eq = np.zeros((1, 2*n))
        A_eq[0, :n] = 1    # alpha_plus coefficients
        A_eq[0, n:] = -1   # alpha_minus coefficients
        b_eq = np.array([0])
        
        constraints.append({
            'type': 'eq',
            'fun': lambda alpha: np.dot(A_eq, alpha) - b_eq,
            'jac': lambda alpha: A_eq
        })
        
        # Bounds: 0 <= alpha_plus <= C*tau, 0 <= alpha_minus <= C*(1-tau)
        bounds = []
        for i in range(n):
            bounds.append((0, self.C * self.tau))        # alpha_plus bounds
        for i in range(n):
            bounds.append((0, self.C * (1 - self.tau)))  # alpha_minus bounds
        
        # Initial guess
        alpha0 = np.zeros(2*n)
        
        # Solve optimization problem
        result = minimize(
            objective, alpha0, method='SLSQP',
            jac=objective_grad, bounds=bounds, constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
        
        alpha_plus = result.x[:n]
        alpha_minus = result.x[n:]
        self.dual_coef_ = alpha_plus - alpha_minus
        
        # Find support vectors (non-zero alphas)
        support_threshold = 1e-6
        self.support_mask_ = (np.abs(self.dual_coef_) > support_threshold)
        self.support_vectors_ = self.X_fit_[self.support_mask_]
        self.support_coef_ = self.dual_coef_[self.support_mask_]
        self.n_support_ = np.sum(self.support_mask_)
        
        return result
    
    def _compute_bias(self, K, y):
        """Compute bias term using KKT conditions"""
        # For SVQR, bias computation is more complex than standard SVR
        # We use support vectors that are not at the bounds
        
        predictions_no_bias = np.dot(K, self.dual_coef_)
        
        # Find support vectors not at bounds for bias computation
        alpha_plus = np.maximum(self.dual_coef_, 0)
        alpha_minus = np.maximum(-self.dual_coef_, 0)
        
        # Support vectors not at upper bounds
        not_at_upper_plus = (alpha_plus < self.C * self.tau - 1e-6) & (alpha_plus > 1e-6)
        not_at_upper_minus = (alpha_minus < self.C * (1 - self.tau) - 1e-6) & (alpha_minus > 1e-6)
        
        bias_candidates = []
        
        if np.any(not_at_upper_plus):
            idx = np.where(not_at_upper_plus)[0]
            for i in idx:
                bias_candidates.append(y[i] - predictions_no_bias[i])
        
        if np.any(not_at_upper_minus):
            idx = np.where(not_at_upper_minus)[0]
            for i in idx:
                bias_candidates.append(y[i] - predictions_no_bias[i])
        
        if bias_candidates:
            self.intercept_ = np.mean(bias_candidates)
        else:
            # Fallback: use median of all residuals
            residuals = y - predictions_no_bias
            self.intercept_ = np.median(residuals)
    
    def fit(self, X, y):
        """Fit the SVQR model"""
        self.X_fit_ = X.copy()
        self.y_fit_ = y.copy()
        
        print(f"Fitting SVQR with tau={self.tau}, C={self.C}, kernel={self.kernel}")
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # Solve dual problem
        self._solve_dual_problem(K, y)
        
        # Compute bias
        self._compute_bias(K, y)
        
        print(f"Training completed. Support vectors: {self.n_support_}/{len(y)}")
        
        return self
    
    def predict(self, X):
        """Predict quantile values"""
        if not hasattr(self, 'dual_coef_'):
            raise ValueError("Model not fitted yet")
        
        K = self._compute_kernel_matrix(X, self.X_fit_)
        predictions = np.dot(K, self.dual_coef_) + self.intercept_
        
        return predictions
    
    def score(self, X, y):
        """Score using pinball loss (lower is better)"""
        y_pred = self.predict(X)
        return -self._pinball_loss(y, y_pred)  # Negative because higher score is better

# ### CLASS ADAM SVQR (PRIMAL PROBLEM)

class AdamSVQR(BaseEstimator, RegressorMixin):
    """
    Support Vector Quantile Regression optimized with Adam Optimizer (Primal Formulation)
    
    This implementation uses Adam optimization algorithm to directly optimize
    the primal problem with weight parameters W and bias b, instead of the 
    dual formulation with alpha parameters.
    
    Uses EXACT kernel features by treating all training points as RBF centers.
    This gives exact kernel representation (not approximation) for small datasets.
    
    Primal Problem:
    minimize: λ||W||²/2 + mean(pinball_loss(y, W^T φ(x) + b))
    where φ(x) = [K(x, x_train[i]) for all i] (EXACT kernel features)
    """
    
    def __init__(self, tau=0.5, C=1.0, epsilon=1e-8, max_iter=1000, 
                 learning_rate=0.001, beta1=0.9, beta2=0.999, tol=1e-6,
                 gamma='auto', use_kernel=True):
        """
        Parameters (mengikuti referensi adam_optimizer.MD):
        -----------
        tau : float, default=0.5
            Quantile level (0 < tau < 1)
        C : float, default=1.0
            Regularization parameter (controls trade-off between regularization and loss)
        epsilon : float, default=1e-8
            ε = 1×10^(-8) untuk stabilitas numerik Adam (sesuai referensi)
        max_iter : int, default=1000
            Maximum iterations untuk optimizer
        learning_rate : float, default=0.001
            δ = 0.001 laju pembelajaran (sesuai referensi adam_optimizer.MD)
        beta1 : float, default=0.9
            λ₁ = 0.9 exponential decay rate untuk momen pertama (sesuai referensi)
        beta2 : float, default=0.999
            λ₂ = 0.999 exponential decay rate untuk momen kedua (sesuai referensi)
        tol : float, default=1e-6
            Convergence tolerance
        gamma : float or 'auto', default='auto'
            RBF kernel bandwidth parameter (only used if use_kernel=True)
        use_kernel : bool, default=True
            If True, use kernel transformation (W will have n_samples dimensions).
            If False, use original features only (W will have n_features dimensions).
            Set to False for simple linear model with interpretable coefficients.
        """
        self.tau = tau
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.tol = tol
        self.gamma = gamma
        self.use_kernel = use_kernel
    
    def _compute_rbf_features(self, X):
        """
        Compute features based on use_kernel setting:
        
        If use_kernel=True:
            Compute EXACT RBF features using all training points as centers
            φ(x) = [K(x, x_train[0]), ..., K(x, x_train[n])]
            
        If use_kernel=False:
            Return original features (no transformation)
            φ(x) = x
        """
        if not self.use_kernel:
            # Linear mode: no kernel transformation
            return X
        
        # Kernel mode: exact RBF features
        if not hasattr(self, 'centers_'):
            # On first call (during training), use input X as centers
            self.centers_ = X.copy()
            
            # Compute gamma
            n_features = X.shape[1]
            if self.gamma == 'auto':
                self.gamma_value_ = 1.0 / n_features
            else:
                self.gamma_value_ = self.gamma
        
        # Compute RBF kernel between X and all centers
        # K(x, c) = exp(-gamma * ||x - c||^2)
        
        # Efficient computation using broadcasting
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x·c
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        centers_norm = np.sum(self.centers_**2, axis=1).reshape(1, -1)
        distances_sq = X_norm + centers_norm - 2 * np.dot(X, self.centers_.T)
        
        # Apply RBF kernel
        features = np.exp(-self.gamma_value_ * distances_sq)
        
        return features
    
    def _pinball_loss(self, y_true, y_pred):
        """Pinball loss function (quantile loss)"""
        error = y_true - y_pred
        return np.mean(np.maximum(self.tau * error, (self.tau - 1) * error))
    
    def _pinball_loss_derivative(self, error):
        """Derivative of pinball loss w.r.t predictions"""
        # d/d(y_pred) pinball_loss = -tau if error >= 0, else -(tau-1)
        return np.where(error >= 0, -self.tau, -(self.tau - 1))
    
    def _compute_objective(self, W, b, X_features, y):
        """
        Compute primal objective function value
        obj = lambda * (1/2)||W||^2 + mean(pinball_loss)
        where lambda = 1/C (regularization strength)
        """
        y_pred = np.dot(X_features, W) + b
        regularization = 0.5 * np.dot(W, W) / self.C  # lambda = 1/C
        loss = self._pinball_loss(y, y_pred)
        return regularization + loss
    
    def _compute_gradients(self, W, b, X_features, y):
        """
        Compute gradients of the primal objective w.r.t W and b
        
        grad_W = W/C + mean(d_pinball * X)
        grad_b = mean(d_pinball)
        """
        n_samples = X_features.shape[0]
        
        # Predictions
        y_pred = np.dot(X_features, W) + b
        
        # Error
        error = y - y_pred
        
        # Derivative of pinball loss w.r.t predictions
        d_pinball = self._pinball_loss_derivative(error)
        
        # Gradient w.r.t W: W/C + (1/n) * sum(d_pinball_i * x_i)
        grad_W = W / self.C + np.dot(X_features.T, d_pinball) / n_samples
        
        # Gradient w.r.t b: (1/n) * sum(d_pinball_i)
        grad_b = np.mean(d_pinball)
        
        return grad_W, grad_b
    
    def _adam_optimizer(self, X_features, y):
        """
        Optimize W and b using Adam optimizer
        Mengikuti algoritma ADAM dari referensi adam_optimizer.MD
        
        Langkah-langkah:
        1. Inisialisasi hyperparameter: lr, beta1, beta2, eps
        2. Inisialisasi parameter W, b = 0
        3. Inisialisasi momen pertama (z) dan momen kedua (v) = 0
        4. Loop untuk setiap iterasi t:
           - Hitung gradien g_w, g_b
           - Update momen pertama z
           - Update momen kedua v  
           - Bias correction untuk z_hat dan v_hat
           - Update W dan b
        
        Returns:
        --------
        n_iter : int
            Number of iterations performed
        """
        n_samples, n_features = X_features.shape
        
        # 1. Hyperparameter sudah di __init__
        # lr, beta1, beta2, eps
        
        # 2. Inisialisasi parameter model
        # w = (0, 0, ..., 0) dengan ukuran 1 x k
        W = np.zeros(n_features)
        b = 0.0
        
        # 3. Inisialisasi momen Adam
        # z_w^(0) = (0, 0, ..., 0) - momen pertama untuk W
        # z_b^(0) = 0 - momen pertama untuk b
        z_W = np.zeros(n_features)
        z_b = 0.0
        
        # v_w^(0) = (0, 0, ..., 0) - momen kedua untuk W
        # v_b^(0) = 0 - momen kedua untuk b
        v_W = np.zeros(n_features)
        v_b = 0.0
        
        # Untuk tracking model terbaik
        best_obj = float('inf')
        best_W = W.copy()
        best_b = b
        patience = 50
        no_improve = 0
        
        prev_obj = float('inf')
        
        # 4-10. Loop iterasi
        for t in range(1, self.max_iter + 1):
            # 4. Menghitung gradien dari fungsi loss
            # g_w^(t) = ∂L/∂w
            # g_b^(t) = ∂L/∂b
            grad_W, grad_b = self._compute_gradients(W, b, X_features, y)
            
            # 5. Memperbaharui perkiraan momen pertama (z)
            # z_w^(t) = β₁ * z_w^(t-1) + (1-β₁) * g_w^(t)
            # z_b^(t) = β₁ * z_b^(t-1) + (1-β₁) * g_b^(t)
            z_W = self.beta1 * z_W + (1 - self.beta1) * grad_W
            z_b = self.beta1 * z_b + (1 - self.beta1) * grad_b
            
            # 6. Memperbaharui perkiraan momen kedua (v)
            # v_w^(t) = β₂ * v_w^(t-1) + (1-β₂) * (g_w^(t) ⊙ g_w^(t))
            # v_b^(t) = β₂ * v_b^(t-1) + (1-β₂) * (g_b^(t) ⊙ g_b^(t))
            # ⊙ adalah produk Hadamard (element-wise)
            v_W = self.beta2 * v_W + (1 - self.beta2) * (grad_W ** 2)
            v_b = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)
            
            # 7. Perbaiki bias pada estimasi momen pertama (ẑ) dan momen kedua (v̂)
            # ẑ_w^(t) = z_w^(t) / (1 - β₁^t)
            # ẑ_b^(t) = z_b^(t) / (1 - β₁^t)
            z_W_hat = z_W / (1 - self.beta1 ** t)
            z_b_hat = z_b / (1 - self.beta1 ** t)
            
            # v̂_w^(t) = v_w^(t) / (1 - β₂^t)
            # v̂_b^(t) = v_b^(t) / (1 - β₂^t)
            v_W_hat = v_W / (1 - self.beta2 ** t)
            v_b_hat = v_b / (1 - self.beta2 ** t)
            
            # 8. Hitung tingkat pembelajaran adaptif (opsional, dalam implementasi kita gunakan lr tetap)
            # Dalam referensi: δ^(t) = (δ^(t-1) * √(1-β₂)) / (1-β₁)
            # Tapi dalam implementasi standar Adam, kita gunakan lr tetap
            
            # 9. Perbarui parameter W dan b
            # w^(t) = w^(t-1) - (lr * ẑ_w^t) / (√(v̂_w^t) + ε)
            # b^(t) = b^(t-1) - (lr * ẑ_b^t) / (√(v̂_b^t) + ε)
            W = W - self.learning_rate * z_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            b = b - self.learning_rate * z_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
            # Compute current objective untuk monitoring
            current_obj = self._compute_objective(W, b, X_features, y)
            
            # Check for improvement
            if current_obj < best_obj - self.tol:
                best_obj = current_obj
                best_W = W.copy()
                best_b = b
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                if t > 100:  # Minimum iterations
                    break
            
            # Convergence check
            if t > 1 and abs(current_obj - prev_obj) < self.tol:
                break
            
            prev_obj = current_obj
        
        # Use best parameters
        self.W_ = best_W
        self.b_ = best_b
        
        return t
    
    def fit(self, X, y):
        """
        Fit the SVQR model using Adam optimizer on primal problem
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Store original training data
        self.X_fit_ = X.copy()
        self.y_fit_ = y.copy()
        
        # Compute RBF features
        X_features = self._compute_rbf_features(X)
        
        print(f"Fitting Adam-SVQR (Primal) with tau={self.tau}, C={self.C}")
        if self.use_kernel:
            print(f"Mode: Kernel - Exact Kernel Features: {X_features.shape[1]} (using all training points as centers)")
        else:
            print(f"Mode: Linear - Original Features: {X_features.shape[1]}")
        
        # Optimize W and b with Adam
        n_iter = self._adam_optimizer(X_features, y)
        
        # Compute final objective and metrics
        final_obj = self._compute_objective(self.W_, self.b_, X_features, y)
        y_pred = self.predict(X)
        final_loss = self._pinball_loss(y, y_pred)
        
        print(f"Training completed in {n_iter} iterations.")
        print(f"Final objective: {final_obj:.6f}, Pinball loss: {final_loss:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Predict quantile values
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        if not hasattr(self, 'W_'):
            raise ValueError("Model not fitted yet")
        
        # Compute RBF features for test data
        X_features = self._compute_rbf_features(X)
        
        return np.dot(X_features, self.W_) + self.b_
    
    def score(self, X, y):
        """Score using negative pinball loss (higher is better)"""
        y_pred = self.predict(X)
        return -self._pinball_loss(y, y_pred)


# ### IMPLEMENTASI SVQR

# Model 1: SVQR dengan SLSQP optimizer (model lama)
print("=" * 60)
print("MODEL 1: AccurateSVQR (SLSQP Optimizer)")
print("=" * 60)
svqr_model_slsqp = AccurateSVQR(tau=0.5, C=1.0, kernel='rbf', gamma='auto')
svqr_model_slsqp.fit(X, y)  # Menggunakan X dengan 2 features
    
y_pred_slsqp = svqr_model_slsqp.predict(X)
mae_slsqp = mean_absolute_error(y, y_pred_slsqp)
pinball_loss_slsqp = svqr_model_slsqp._pinball_loss(y, y_pred_slsqp)
    
print(f"MAE: {mae_slsqp:.4f}")
print(f"Pinball Loss (τ=0.5): {pinball_loss_slsqp:.4f}")

# Model 2: SVQR dengan Adam optimizer (model baru - Primal Problem + Exact Kernel)
print("\n" + "=" * 60)
print("MODEL 2: AdamSVQR (Adam Optimizer - Primal + Exact Kernel)")
print("=" * 60)
# Gunakan exact kernel features (bukan aproksimasi RFF)
svqr_model = AdamSVQR(tau=0.5, C=1.0, learning_rate=0.001, max_iter=2000,
                      gamma='auto')
svqr_model.fit(X, y)  # Menggunakan X dengan 2 features
    
y_pred = svqr_model.predict(X)
mae = mean_absolute_error(y, y_pred)
pinball_loss = svqr_model._pinball_loss(y, y_pred)
    
print(f"MAE: {mae:.4f}")
print(f"Pinball Loss (τ=0.5): {pinball_loss:.4f}")

# Print parameter W dan b (untuk kernel mode, W ada 100 nilai)
print(f"\nParameter W (weights): shape={svqr_model.W_.shape}")
print(f"First 10 values of W: {svqr_model.W_[:10]}")
print(f"Parameter b (bias): {svqr_model.b_:.6f}")
print(f"W statistics - Min: {svqr_model.W_.min():.6f}, Max: {svqr_model.W_.max():.6f}, Mean: {svqr_model.W_.mean():.6f}")

# Model 3: Mode LINEAR (tanpa kernel) - untuk interpretasi langsung
print("\n" + "=" * 60)
print("MODEL 3: AdamSVQR (Linear Mode - No Kernel)")
print("=" * 60)
svqr_linear = AdamSVQR(tau=0.5, C=1.0, learning_rate=0.001, max_iter=2000,
                       use_kernel=False)  # Nonaktifkan kernel!
svqr_linear.fit(X, y)  # Menggunakan X dengan 2 features

y_pred_linear = svqr_linear.predict(X)
mae_linear = mean_absolute_error(y, y_pred_linear)
pinball_loss_linear = svqr_linear._pinball_loss(y, y_pred_linear)

print(f"MAE: {mae_linear:.4f}")
print(f"Pinball Loss (τ=0.5): {pinball_loss_linear:.4f}")
print(f"\n>>> INTERPRETABLE PARAMETERS <<<")
print(f"W[0] (effect of x1 on y): {svqr_linear.W_[0]:.6f}")
print(f"W[1] (effect of x2 on y): {svqr_linear.W_[1]:.6f}")
print(f"W[2] (effect of x3 on y): {svqr_linear.W_[2]:.6f}")
print(f"W[3] (effect of x4 on y): {svqr_linear.W_[3]:.6f}")
print(f"b (intercept): {svqr_linear.b_:.6f}")
print(f"\nModel: y = {svqr_linear.W_[0]:.6f}*x1 + {svqr_linear.W_[1]:.6f}*x2 + {svqr_linear.W_[2]:.6f}*x3 + {svqr_linear.W_[3]:.6f}*x4 + {svqr_linear.b_:.6f}")

# Perbandingan
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"MAE Improvement: {((mae_slsqp - mae) / mae_slsqp * 100):.2f}%")
print(f"Pinball Loss Improvement: {((pinball_loss_slsqp - pinball_loss) / pinball_loss_slsqp * 100):.2f}%")

# ### GAMBAR

plt.figure(figsize=(12, 5))

# Plot 1: SLSQP (gunakan x1 sebagai x-axis)
plt.subplot(1, 2, 1)
plt.scatter(x1, y, alpha=0.6, s=20, label='Data Asli')
plt.plot(x1, y_pred_slsqp, 'r-', linewidth=2, label='SVQR-SLSQP (τ=0.5)')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('AccurateSVQR (SLSQP Optimizer)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Adam (gunakan x1 sebagai x-axis)
plt.subplot(1, 2, 2)
plt.scatter(x1, y, alpha=0.6, s=20, label='Data Asli')
plt.plot(x1, y_pred, 'g-', linewidth=2, label='AdamSVQR (τ=0.5)')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('AdamSVQR (Adam Optimizer)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()  # Menampilkan gambar

# ### DATA BANGKITAN NORMAL INDEPENDEN
