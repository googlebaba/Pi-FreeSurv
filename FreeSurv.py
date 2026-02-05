import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import heapq
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
class FreeSurv:
    def __init__(self, alpha=1.0, max_iter=100, tol=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        """
        Initialize the FreeSurv feature selector.
        
        Parameters:
        - alpha: Regularization parameter (corresponds to lambda1), controls sparsity.
        - max_iter: Maximum number of iterations for coordinate descent.
        - tol: Convergence threshold.
        - device: Computation device ('cpu' or 'cuda').
        - verbose: Whether to print progress bars.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.verbose = verbose
        
        # Core result: the optimized sparse weight vector z
        self.z_ = None 
        # Auxiliary results
        self.selected_indices_ = None
        self.feature_importances_ = None

    def _rbf_kernel(self, X, sigma=None):
        """Compute the RBF (Radial Basis Function) kernel matrix."""
        XX = X @ X.t()
        n = X.shape[0]
        X_sqnorms = torch.diag(XX)
        R = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        
        if sigma is None:
            # Heuristic estimation of sigma using the median
            valid_R = R[R > 0]
            if valid_R.numel() > 0:
                sigma = torch.sqrt(torch.median(valid_R))
            else:
                sigma = torch.tensor(1.0).to(self.device)
                
        K = torch.exp(-R / (2 * sigma * sigma))
        return K

    def _compute_hsic(self, X_tensor, Y_tensor):
        """Compute HSIC (Hilbert-Schmidt Independence Criterion) for feature-feature correlation."""
        n = X_tensor.shape[0]
        K = self._rbf_kernel(X_tensor)
        L = self._rbf_kernel(Y_tensor)
        H = torch.eye(n).to(self.device) - 1 / n
        hsic_val = torch.trace(K @ H @ L @ H)
        return hsic_val

    def _compute_matrix_A(self, time_vector):
        """Compute the risk set matrix A for the Cox model."""
        n = len(time_vector)
        time_vector = time_vector.reshape(n, 1)
        # Generate comparison matrix (Indicator matrix for risk set: T_j >= T_i)
        comparison_matrix = torch.tensor(time_vector <= time_vector.T, dtype=torch.float32).to(self.device)
        counts = comparison_matrix.sum(axis=1, keepdim=True)
        A = comparison_matrix / counts
        return A

    def _hsic_cox(self, X_col, T, C, A, sigma=None):
        """Compute HSIC-Cox value for feature-target correlation."""
        K = self._rbf_kernel(X_col)
        
        # Kernel matrix calculation for Time
        XX_T = T @ T.t()
        T_sqnorms = torch.diag(XX_T)
        R_T = -2 * XX_T + T_sqnorms.unsqueeze(1) + T_sqnorms.unsqueeze(0)
        
        if sigma is None:
            valid_R = R_T[R_T > 0]
            sigma = torch.sqrt(torch.median(valid_R)) if valid_R.numel() > 0 else 1.0
            
        L = torch.exp(-R_T / (2 * sigma * sigma))
        
        # Incorporate censoring information
        delta = torch.mm(C, C.t())
        delta_L = torch.mul(L, delta)
        
        H = torch.eye(T.shape[0]).to(self.device) - A
        hsic_val = torch.trace(H.t() @ delta_L @ H @ K)
        return hsic_val

    def _coordinate_descent(self, yx_HSIC, xx_HSIC):
        n_samples = yx_HSIC.shape[0]
        n_features = xx_HSIC.shape[0]
        alpha = self.alpha
        term1_matrix = yx_HSIC
        term2_matrix = xx_HSIC
        z = torch.zeros(n_features).to(self.device)
        gap = 0.0
        converged = 0.0
        d_w_tol = self.tol
        #tol = tol * y.pow(2).sum()  # [N,]

        # compute squared norms of the columns of X
        #norm_cols_X = X.pow(2).sum(0)  # [K,]
        alpha = alpha * n_samples

        # initialize residual
        best_val_c_index = 0
        val_c_index = 0
        top_N = 20
        for n_iter in range(self.max_iter):
            #R_norm2 = R.pow(2).sum()
            #if converged:
            #    break
            z_max = torch.tensor(0.0).to(self.device)
            d_z_max = torch.tensor(0.0).to(self.device)
            for i in range(n_features):  # Loop over components
                if term2_matrix[i , i] == 0:
                    continue

                atom_i = term2_matrix[i,:]

                z_c = z.clone()
                z_i = z[i].clone()
                z_c[i] = 0
                tho = term1_matrix[i] -0.5 * atom_i.reshape((1, -1)) @ z_c.reshape((-1, 1))
                if tho < 0.0:
                    z[i] = 0.0
                else:
                    #z[i] = F.softshrink(R.reshape(1, -1).matmul(atom_i.reshape(-1, 1)), alpha)
                    z[i] = F.softshrink(tho, alpha)
                    z[i] /= (term2_matrix[i , i])
                #print('z[i]', z[i])
                z_new_i = z[i]

                # update the maximum absolute coefficient update
                d_z_max = torch.maximum(d_z_max, (z_new_i - z_i).abs())
                z_max = torch.maximum(z_max, z_new_i.abs())
            
            #with torch.no_grad():
            #    tmp = z.reshape(-1)
            #    indices = torch.topk(abs(tmp), top_N)[1].cpu().numpy()

        return z.reshape((-1,))

    def _mark_top_n_features(self, input_list, n=4):                                                      
        """                                                                                               
        Marks the positions of the top N largest values in a Python list as 1,                            
        and the rest as 0.                                                                                
                                                                                                          
        Args:                                                                                             
        input_list (list): The input Python list.                                                         
        n (int): The number of largest values to mark. Defaults to 4.                                     
                                                                                                          
        Returns:                                                                                          
        tuple: A tuple containing:
               - list: A list of the same length as the input, with 1s at the indices of 
                       the top N values and 0s elsewhere.
               - list: A list of the indices corresponding to the top N values.
        """                                                                                               
        if not isinstance(input_list, list):                                                              
            raise TypeError("Input must be a Python list.")                                              
        if not input_list: # Empty list                                                                       
            return []                                                                                     
        if n <= 0: # If n is less than or equal to 0, mark all as 0                                                    
            return [0] * len(input_list)                                                                  
        if n >= len(input_list): # If n is greater than or equal to list length, mark all as 1                                
            return [1] * len(input_list)                                                                  
                                                                                                          
        # 1. Pair list elements with their original indices                                                                   
        # indexed_list will be [(value, original_index), ...]                                                
        indexed_list = [(value, i) for i, value in enumerate(input_list)]                                 
                                                                                                          
        # 2. Use heapq.nlargest to find the top N values and their original indices                                            
        # key=lambda x: x[0] tells nlargest to compare based on the first element (the value)                               
        top_n_pairs = heapq.nlargest(n, indexed_list, key=lambda x: x[0])                                 
                                                                                                          
        # 3. Extract the original indices of these top values                                                                     
        # Using a set improves lookup efficiency, especially for large lists                                               
        top_n_indices = {pair[1] for pair in top_n_pairs}                                                 
                                                                                                          
        # 4. Create a result list of zeros with the same length as the original list                                                     
        result_list = [0] * len(input_list)                                                               
                                                                                                          
        # 5. Iterate through the range of the list and set the corresponding indices to 1                                                     
        for i in range(len(input_list)):                                                                  
            if i in top_n_indices:                                                                        
                result_list[i] = 1                                                                        
                                                                                                          
        return result_list, list(top_n_indices)



    def fit(self, X, duration, event):
        """
        Fit the model to select features.
        
        Parameters:
        - X: Feature matrix (numpy array or pandas DataFrame).
        - duration: Time to event vector.
        - event: Event indicator vector (0=censored, 1=event).
        
        Returns:
        - self.z_: Optimized weight vector (numpy array).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Convert inputs to tensors
        scaler = StandardScaler()
        norm_duration = scaler.fit_transform(duration.reshape((-1, 1)))
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        T_tensor = torch.tensor(norm_duration, dtype=torch.float32).reshape(-1, 1).to(self.device)
        C_tensor = torch.tensor(event, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        n_samples, n_features = X_tensor.shape
        
        # 1. Compute matrix A for Cox partial likelihood
        A_tensor = self._compute_matrix_A(duration) 
        
        # 2. Compute yx_HSIC (Feature-Target Correlation Vector)
        yx_HSIC_list = []
        iter_cols = range(n_features)
        #if self.verbose:
        #    iter_cols = tqdm(iter_cols, desc="Computing Feature-Target HSIC")
            
        for i in iter_cols:
            X_col = X_tensor[:, i].reshape(-1, 1)
            yx_HSIC_list.append(self._hsic_cox(X_col, T_tensor, C_tensor, A_tensor))
        yx_HSIC = torch.tensor(yx_HSIC_list).to(self.device).reshape(-1, 1)
        
        # 3. Compute xx_HSIC (Feature-Feature Correlation Matrix)
        # Initialize zero matrix
        xx_HSIC = torch.zeros((n_features, n_features)).to(self.device)
        iter_cols = range(n_features)
        #if self.verbose:
        #    iter_cols = tqdm(iter_cols, desc="Computing Feature-Feature HSIC")
            
        for i in iter_cols:
            X_i = X_tensor[:, i].reshape(-1, 1)
            # Diagonal elements
            xx_HSIC[i, i] = self._compute_hsic(X_i, X_i)
            
            # Fill lower triangle (symmetric matrix) to save computation
            for j in range(i):
                X_j = X_tensor[:, j].reshape(-1, 1)
                val = self._compute_hsic(X_i, X_j)
                xx_HSIC[i, j] = val
                xx_HSIC[j, i] = val
        
        # 4. Solve for z using Coordinate Descent
        z_tensor = self._coordinate_descent(yx_HSIC, xx_HSIC)
        
        # 5. Save results (convert to numpy for compatibility)
        self.z_ = z_tensor.cpu().detach().numpy()
        self.feature_importances_ = np.abs(self.z_)
        self.y_pred_binary, self.selected_indices_ = self._mark_top_n_features(list(self.feature_importances_)) # Descending order
        count_zero = len(list(self.feature_importances_))-list(self.feature_importances_).count(0)
        print(f"[FreeSurv] Number of features with non-zero coefficients: {count_zero}")
        # Return z directly for convenience
        return self.z_


    def get_z(self):
        """Explicitly retrieve the weight vector z."""
        if self.z_ is None:
            raise RuntimeError("Model not fitted yet. Call 'fit' first.")
        return self.z_

    def evaluate_fs(self, y_true_feature_labels):
        """
        Returns a dictionary containing sensitivity and specificity, or None if not calculated.
        """
        self.sensitivity_ = recall_score(y_true_feature_labels, self.y_pred_binary, pos_label=1)
        self.specificity_ = recall_score(y_true_feature_labels, self.y_pred_binary, pos_label=0)
        
        #if self.verbose:
        #    print(f"[FreeSurv Sensitivity: {self.sensitivity_:.4f}, Specificity: {self.specificity_:.4f}")


        return {'sensitivity': self.sensitivity_, 'specificity': self.specificity_}


    def transform(self, X, top_n=None):
        """
        Transform data by selecting top features based on learned weights.
        
        Parameters:
        - top_n: Number of features to keep. If None, keep all features with non-zero weights.
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Model not fitted yet.")
            
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        if top_n is None:
            # Return features with non-zero weights
            mask = self.feature_importances_ > 0
            return X_np[:, mask]
        else:
            # Return top N features
            indices = self.selected_indices_[:top_n]
            return X_np[:, indices]
    
    def get_selected_feature_indices(self):
        """
        Returns the indices of the top N (or all non-zero) selected features.
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Estimator not fitted yet. Call 'fit' first.")
        return self.selected_indices_
           
    def get_support(self, top_n=30):
        """Return indices of the selected features."""
        if self.selected_indices_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.selected_indices_[:top_n]

# ================= Usage Example =================
if __name__ == "__main__":
    # Simulated Data
    # X = np.random.rand(100, 20)
    # T = np.random.uniform(10, 50, 100)
    # E = np.random.randint(0, 2, 100)
    
    # Initialize and fit
    # model = FreeSurv(alpha=0.1, device='cpu')
    # z = model.fit(X, T, E)
    
    # print("Optimized z vector:", z)
    # print("Top 5 feature indices:", model.get_support(top_n=5))
    pass

