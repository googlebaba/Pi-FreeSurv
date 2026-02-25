import numpy as np
import pandas as pd
from skglm.datafits import Cox
from skglm.penalties import L1, L1_plus_L2
from skglm.solvers import AndersonCD, ProxNewton # Assuming AndersonCD as the solver based on the snippet
from skglm.utils.jit_compilation import compiled_clone
from sklearn.metrics import recall_score
import sys
from Simulated_data_generation import SurvivalDataGenerator

import heapq
try:
    from skglm.utils.data_manipulation import make_correlated_data
except ImportError:
    # Fallback for older skglm versions or different environments
    # This might not be strictly needed for the class itself, but good for completeness
    pass

class CoxFeatureSelector:
    """
    A class to perform feature selection using Cox regression with L1 or ElasticNet penalties
    via the skglm library. It can output top-N feature indices and evaluate
    selection performance (sensitivity, specificity) against ground truth feature labels.
    """
    def __init__(self,
                 lambda1_l1=0.1,         # Alpha for L1 penalty
                 alpha_elastic=0.1,      # Alpha for ElasticNet penalty
                 l1_ratio_elastic=0.5,   # L1_ratio for ElasticNet penalty (between 0 and 1)
                 solver_tol=1e-4,        # Tolerance for the skglm solver
                 solver_max_iter=50,   # Max iterations for the skglm solver
                 coef_threshold=1e-6,    # Threshold to consider a coefficient non-zero
                 verbose=False):         # Whether to print solver details

        self.lambda1_l1 = lambda1_l1
        self.alpha_elastic = alpha_elastic
        self.l1_ratio_elastic = l1_ratio_elastic
        self.solver_tol = solver_tol
        self.solver_max_iter = solver_max_iter
        self.coef_threshold = coef_threshold
        self.verbose = verbose

        self.solver = ProxNewton(fit_intercept=False, max_iter=500)

        # Attributes to store results after fitting
        self.coef_ = None              # Absolute value of estimated coefficients
        self.num_selected_features_ = None # Number of features with non-zero coefficients
        self.top_indices_ = None       # Indices of top N features
        self.sensitivity_ = None       # Sensitivity of feature selection
        self.specificity_ = None       # Specificity of feature selection
        self.penalty_type_ = None      # Type of penalty used ('L1' or 'elastic')

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


    def fit(self, X, duration, event, penalty_type='L1', top_n=None, y_true_feature_labels=None):
        """
        Fits the Cox regression model with the specified penalty and identifies features.

        Parameters:
        - X (np.ndarray): Feature matrix (n_samples, n_features).
        - duration (np.ndarray): Array of event/censoring times.
        - event (np.ndarray): Array of event indicators (1 for event, 0 for censored).
        - penalty_type (str): Type of penalty to use, 'L1' or 'elastic'.
        - top_n (int, optional): Number of top features to select. If None,
                                 all non-zero features (above coef_threshold) are considered.
                                 This is also used for sensitivity/specificity calculation.
        - y_true_feature_labels (np.ndarray, optional): Ground truth binary labels for features
                                                    (1 for truly relevant, 0 for irrelevant).
                                                    Required to calculate sensitivity and specificity.

        Returns:
        - self: The fitted estimator.
        """
        if penalty_type not in ['L1', 'elastic']:
            raise ValueError("penalty_type must be 'L1' or 'elastic'.")
        
        if top_n is not None and (not isinstance(top_n, int) or top_n <= 0):
            raise ValueError("top_n must be a positive integer or None.")

        self.penalty_type_ = penalty_type

        # Prepare y for skglm (must be a 2D array [durations, events])
        y_skglm = np.concatenate((np.array(duration).reshape((-1, 1)), np.array(event).reshape((-1, 1))), axis=1)
        # Select penalty
        if self.penalty_type_ == 'L1':
            penalty = L1(alpha=self.lambda1_l1)
        else: # 'elastic'
            penalty = L1_plus_L2(alpha=self.alpha_elastic, l1_ratio=self.l1_ratio_elastic)

        # Initialize datafit and penalty with compiled_clone for skglm
        datafit_instance = compiled_clone(Cox())
        penalty_instance = compiled_clone(penalty)

        # Initialize datafit with data
        X=np.array(X)
        datafit_instance.initialize(X, y_skglm)

        # Solve the problem
        # solver.solve returns (w, Xw, E, C_out)
        w_sk, _, _ = self.solver.solve(X, y_skglm, datafit_instance, penalty_instance)

        # Store absolute coefficients
        self.coef_ = np.abs(w_sk)

        # Count selected features based on coefficient threshold
        self.num_selected_features_ = np.sum(self.coef_ > 0)
        if self.verbose:
            print(f"[{self.penalty_type_}] Number of features with non-zero coefficients: {self.num_selected_features_}")

        # Determine top N indices
            
        # Calculate Sensitivity and Specificity if ground truth labels are provided
        if y_true_feature_labels is not None:
            if len(y_true_feature_labels) != X.shape[1]:
                raise ValueError("Length of y_true_feature_labels must match number of features in X.")
            
            if top_n is None:
                # If top_n is not specified, sensitivity/specificity based on non-zero coeffs
                self.y_pred_binary = (self.coef_ > self.coef_threshold).astype(int)
            else:
                # If top_n is specified, use the top_n features for metrics
                self.y_pred_binary, self.top_indices_ = self._mark_top_n_features(list(self.coef_), top_n)

        return self

    def get_coefficients(self):
        """Returns the absolute value of the estimated coefficients."""
        if self.coef_ is None:
            raise RuntimeError("Estimator not fitted yet. Call 'fit' first.")
        return self.coef_

    def get_selected_feature_indices(self):
        """
        Returns the indices of the top N (or all non-zero) selected features.
        """
        if self.top_indices_ is None:
            raise RuntimeError("Estimator not fitted yet. Call 'fit' first.")
        return self.top_indices_

    def get_selection_metrics(self, y_true_feature_labels):
        """
        Returns a dictionary containing sensitivity and specificity, or None if not calculated.
        """

        self.sensitivity_ = recall_score(y_true_feature_labels, self.y_pred_binary, pos_label=1)
        self.specificity_ = recall_score(y_true_feature_labels, self.y_pred_binary, pos_label=0)

        if self.sensitivity_ is None:
            print("Warning: Sensitivity and Specificity not calculated. y_true_feature_labels was not provided to fit().")
            return None
        return {'sensitivity': self.sensitivity_, 'specificity': self.specificity_}


