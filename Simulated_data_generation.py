import numpy as np
import math
from FreeSurv import FreeSurv
import heapq
from sklearn.metrics import recall_score

class SurvivalDataGenerator:
    def __init__(self, seed=None):
        """
        Initializes the data generator.
        :param seed: Random seed for reproducibility.
        """
        self.threshold = 700  # Threshold to prevent exponential overflow in g(x) calculations

    def _gen_cov_matrix(self, p, start, end):
        """
        Generates a p x p covariance matrix with random intra-group correlations.
        """
        covariance_matrix = np.diag([1] * p).astype(float)
        # Sets correlation only for the first 3 features (referencing original code's range(3) logic)
        limit = min(p, 3)
        for i in range(limit):
            for j in range(i + 1, limit):
                intra_corr = np.random.uniform(start, end)
                covariance_matrix[i, j] = intra_corr
                covariance_matrix[j, i] = intra_corr
        
        # Ensures the matrix is positive semi-definite
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        min_eigenvalue = np.min(eigenvalues.real)
        if min_eigenvalue < 0.01:
            # Note: original code used n_features here, assuming p is the intended dimension
            covariance_matrix = covariance_matrix + np.eye(p) * (0.01 - min_eigenvalue)
        return covariance_matrix

    def _cal_gx_additive(self, S1, S2, S3, S4):
        """Calculates the non-linear function g(x) for the additive mode."""
        # Original logic: exp(S1^2) + S2^2 + S3^2 + exp(S4)
        return np.exp(S1**2) + S2**2 + S3**2 + np.exp(S4)

    def _cal_gx_nonadditive(self, S1, S2, S3, S4):
        """Calculates the non-linear function g(x) for the non-additive (interactive) mode."""
        # Original logic: S1^2 * exp(S2^2) + S3^2 * exp(S4)
        return S1**2 * np.exp(S2**2) + S3**2 * np.exp(S4)

    def _add_censoring(self, Y, X, ind, n_samples, correlated=False):
        """
        Adds censoring time C and computes observed time Y_obs and event indicator E.
        """
        Y = Y.reshape((-1, 1))
        
        if ind:
            # --- Independent Censoring ---
            C = np.random.exponential(scale=5, size=n_samples)
            if correlated:
                cut_off = np.percentile(Y,80)
                cut_off = math.floor(math.log10(abs(cut_off)))
                C = np.random.exponential(scale=10**(cut_off), size=n_samples)

        else:
            # --- Dependent Censoring ---
            # C depends on features X. Original code logic: C = exp(0.1*tmp + noise)
            # Uses the first few columns of X to build the dependency.
            # Note: Ensure X has at least 3 columns for full original logic to apply.
            cols = min(X.shape[1], 3)
            S_sub = X[:, :cols]
            
            # Constructs a non-linear combination as the basis for the censoring mechanism.
            # tmp = S[:, 0]**2 + exp(2*S[:, 1]) + 3*S[:, 2]**2 (if enough columns are present)
            tmp = S_sub[:, 0]**2
            if cols > 1: tmp += np.exp(2 * S_sub[:, 1])
            if cols > 2: tmp += 3 * S_sub[:, 2]**2
            
            noise = np.random.normal(0, 0.5, size=n_samples)
            C = np.exp(0.1 * tmp + noise)

        C = C.reshape((-1, 1))
        
        # Calculate observed time and event indicator
        # event = 1 if Y <= C (event occurred), event = 0 if Y > C (censored)
        # Note typical definition: T_obs = min(T, C), delta = I(T <= C)
        indicator = (Y < C).astype(int) 
        Y_obs = np.minimum(Y, C)
        
        censoring_rate = 1 - np.sum(indicator) / n_samples
        #print(f"Censoring rate: {censoring_rate:.4f}")
        
        return Y_obs, indicator, X

    def generate_correlated_data(self, n_samples, begin, end, mode="Cox-additive", ind=True):
        """
        Generates data with correlated features.
        Corresponds to the part in the original code involving gen_Cov and C1, C2...
        """
        # 1. Generate covariance matrices for feature groups
        Sigma_s1 = self._gen_cov_matrix(3, begin, end)
        Sigma_s2 = self._gen_cov_matrix(3, begin, end)
        Sigma_s3 = self._gen_cov_matrix(3, begin, end)
        Sigma_s4 = self._gen_cov_matrix(3, begin, end)

        # 2. Generate multivariate normal features for each group
        C1 = np.random.multivariate_normal([0]*3, Sigma_s1, n_samples)
        C2 = np.random.multivariate_normal([0]*3, Sigma_s2, n_samples)
        C3 = np.random.multivariate_normal([0]*3, Sigma_s3, n_samples)
        C4 = np.random.multivariate_normal([0]*3, Sigma_s4, n_samples)

        # Extract key variables from each group (primarily used for generating Y)
        S1, S11, S12 = C1[:, 0], C1[:, 1], C1[:, 2]
        S2, S21, S22 = C2[:, 0], C2[:, 1], C2[:, 2]
        S3, S31, S32 = C3[:, 0], C3[:, 1], C3[:, 2]
        S4, S41, S42 = C4[:, 0], C4[:, 1], C4[:, 2]

        U = np.random.uniform(0, 1, size=n_samples)

        # 3. Generate true survival time Y based on the specified mode
        if mode == "Cox-additive":
            # Cox model: h(t|x) = h0(t) * exp(g(x))
            # Assuming h0(t) = lambda (constant baseline hazard, e.g., exponential distribution)
            # T = -log(U) / (lambda * exp(g(x)))
            # Original code logic for g(x): S1^2 + exp(S2^2) + 2*S3^2 + exp(S4^2)
            # Assuming baseline hazard rate lambda = 0.5
            g_x = S1**2 + np.exp(S2**2) + 2 * S3**2 + np.exp(S4**2)
            Y = -np.log(U) / (0.5 * np.exp(g_x))

        elif mode == "Cox-nonadditive":
            g_x = S1**2 * np.exp(S2**2) + S3**2 + np.exp(S4**2)
            Y = -np.log(U) / (0.5 * np.exp(g_x))

        elif mode == "log_T-additive":
            # Accelerated Failure Time (AFT) model: log(T) = g(x) + error
            # Original code used _cal_gx_additive: exp(S1^2) + S2^2 + S3^2 + exp(S4)
            g_x = self._cal_gx_additive(S1, S2, S3, S4)
            g_x = np.where(g_x > self.threshold, self.threshold, g_x) # Truncate to prevent overflow
            noise = np.random.normal(0, 0.5, size=n_samples)
            Y = np.exp(g_x + noise)

        elif mode == "log_T-nonadditive":
            # Original code used _cal_gx_nonadditive: S1^2 * exp(S2^2) + S3^2 * exp(S4)
            g_x = self._cal_gx_nonadditive(S1, S2, S3, S4)
            g_x = np.where(g_x > self.threshold, self.threshold, g_x)
            noise = np.random.normal(0, 0.5, size=n_samples)
            Y = np.exp(g_x + noise)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 4. Combine all features for output
        # Shape of S: (n_samples, 12)
        S = np.concatenate([
            S11.reshape(-1,1), S12.reshape(-1,1), 
            S21.reshape(-1,1), S22.reshape(-1,1), 
            S31.reshape(-1,1), S32.reshape(-1,1), 
            S41.reshape(-1,1), S42.reshape(-1,1),
            S1.reshape(-1,1), S2.reshape(-1,1), S3.reshape(-1,1), S4.reshape(-1,1)
        ], axis=1)

        # 5. Add censoring
        return self._add_censoring(Y, S, ind, n_samples, True)

    def generate_uncorrelated_data(self, n_fea, n_samples, mode="Cox-additive", ind=True):
        """
        Generates data with uncorrelated features.
        Corresponds to the part in the original code directly generating S (multivariate_normal with eye).
        """
        # 1. Generate independent normal features
        S = np.random.multivariate_normal([0]*n_fea, np.eye(n_fea), n_samples)
        U = np.random.uniform(0, 1, size=n_samples)

        # Assuming the last 4 features are key features (inferred from original code's S[:, -1] indexing)
        S_k1, S_k2, S_k3, S_k4 = S[:, -1], S[:, -2], S[:, -3], S[:, -4]

        # 2. Generate true survival time Y based on the specified mode
        if mode == "Cox-additive":
            # Original code: tmp = -2 * np.sin(2*S1) + S2**2 + S3 + np.exp(-S4)
            # Here S1...S4 correspond to the S_k1...S_k4 keys
            g_x = -2 * np.sin(2*S_k1) + S_k2**2 + S_k3 + np.exp(-S_k4)
            # Y = -log(U) / (0.5 * exp(g(x)))
            # Original code: tmp=exp(-g_x), Y = -log(U)*tmp / 0.5 => equivalent
            Y = -np.log(U) / (0.5 * np.exp(g_x))

        elif mode == "Cox-nonadditive": # Original code spelled as Cox-non-additive
            # Original code: tmp = S[:,-1] * exp(2*S[:,-2]) + S[:,-3]^2 * exp(S[:,-4])
            g_x = S_k1 * np.exp(2*S_k2) + S_k3**2 * np.exp(S_k4)
            Y = -np.log(U) / (0.5 * np.exp(g_x))

        elif mode == "log_T-additive":
            g_x = -2 * np.sin(2*S_k1) + S_k2**2 + S_k3 + np.exp(-S_k4)
            noise = np.random.normal(0, 0.5, size=n_samples)
            Y = np.exp(g_x + noise)

        elif mode == "log_T-nonadditive": # Original code spelled as log_T-non-additive
            g_x = S_k1 * np.exp(2*S_k2) + S_k3**2 * np.exp(S_k4)
            noise = np.random.normal(0, 0.5, size=n_samples)
            Y = np.exp(g_x + noise)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 3. Add censoring

        true_relevant_features_idx = [-1, -2, -3, -4]
        return self._add_censoring(Y, S, ind, n_samples)


