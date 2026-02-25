from Baselines import CoxFeatureSelector
import argparse
import numpy as np
from TreeSurvival import RFSurvival
import pandas as pd
from tqdm import tqdm
from FreeSurv import FreeSurv
from Data_loader import Data
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler # Added import for StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Assume the following classes are defined: Data, RFSurvival, FreeSurv, CoxFeatureSelector

def evaluate_with_bootstrap(model, df_train, df_test, duration_col, event_col, times, n_bootstraps=1000, seed=None):
    """
    Helper function: Computes mean and standard deviation of C-index and AUC using Bootstrap.
    Adapts to RFSurvival(df, duration_col, event_col) calling convention.
    """
    if seed: np.random.seed(seed)

    # 1. Train the model (on the full training dataset)
    # df_train here already contains filtered feature columns + time column + event column
    model.fit(df_train, duration_col=duration_col, event_col=event_col)

    # 2. Prepare structured arrays in sksurv format (for AUC calculation)
    # y_train_struct: Reference for cumulative_dynamic_auc
    y_train_struct = Surv.from_arrays(
        event=df_train[event_col].astype(bool).values,
        time=df_train[duration_col].values
    )

    # 3. Define a single Bootstrap task
    def single_bootstrap_run(i):
        # Sample with replacement from the test set indices
        n_test = len(df_test)
        indices = np.random.choice(df_test.index, size=n_test, replace=True)

        # Construct the Bootstrap test set DataFrame
        df_test_boot = df_test.loc[indices]

        # Construct the corresponding structured labels (for AUC calculation)
        y_test_boot_struct = Surv.from_arrays(
            event=df_test_boot[event_col].astype(bool).values,
            time=df_test_boot[duration_col].values
        )

        # --- Calculate C-index ---
        try:
            # The score method accepts a DataFrame
            c_index = model.score(df_test_boot, duration_col=duration_col, event_col=event_col)
        except Exception:
            c_index = np.nan

        # --- Calculate AUC ---
        try:
            # Assume model.model is the underlying RandomForestSurvival instance
            # predict_cumulative_hazard_function typically only accepts the feature matrix (without labels)
            X_test_boot_only = df_test_boot.drop(columns=[duration_col, event_col])
            # model.model here points to the internal sksurv model
            preds = model.model.predict_cumulative_hazard_function(X_test_boot_only)
            chf_at_times = np.row_stack([fn(times) for fn in preds])

            auc, mean_auc = cumulative_dynamic_auc(y_train_struct, y_test_boot_struct, chf_at_times, times)
            auc_val = mean_auc
        except Exception:
            auc_val = np.nan

        return c_index, auc_val

    # 4. Run Bootstrap in parallel
    results = Parallel(n_jobs=-1)(delayed(single_bootstrap_run)(i) for i in range(n_bootstraps))

    # 5. Result statistics (filter out NaNs)
    c_indices = [r[0] for r in results if not np.isnan(r[0])]
    aucs = [r[1] for r in results if not np.isnan(r[1])]

    return {
        'C-index_mean': np.mean(c_indices) if c_indices else 0,
        'C-index_std': np.std(c_indices) if c_indices else 0,
        'AUC_mean': np.mean(aucs) if aucs else 0,
        'AUC_std': np.std(aucs) if aucs else 0
    }

def run_HCC_experiments(l1_lambda, time_points_auc, elastic_lambda, FreeSurv_lambda, seed=42):
    if seed is not None: np.random.seed(seed)

    # 1. Data Loading
    duration_col = "Survival.months"
    event_col = "Survival.status"

    data_loader = Data(data_name="HCC", duration_col = "Survival.months", event_col = "Survival.status")
    datasets = {}
    # Load train, test1, test2
    for set_name in ["train", "test1", "test2"]:
        Y, E, X = data_loader.get_data(data_name="HCC", set_name=set_name, duration_col=duration_col, event_col=event_col)

        # Convert to DataFrame and merge labels
        df = pd.DataFrame(X)
        # Record feature column names (assuming integer indices)
        feature_cols = df.columns.tolist()

        df[duration_col] = Y
        df[event_col] = E

        datasets[set_name] = {
            'df': df,
            'X_vals': X, # Original numerical matrix, for feature selector fit
            'Y_vals': Y,
            'E_vals': E
        }

    # AUC calculation time point (e.g., median of training event times)
    #train_events_time = datasets['train']['Y_vals'][datasets['train']['E_vals'] == 1]
    time_points = time_points_auc # np.percentile(train_events_time, [50])

    # 2. Define Feature Selection Models
    selectors = {
        'L1': (
            CoxFeatureSelector(lambda1_l1=l1_lambda, verbose=True),
            {'penalty_type': 'L1'}
        ),
        'ElasticNet': (
            CoxFeatureSelector(alpha_elastic=elastic_lambda, l1_ratio_elastic=0.5, verbose=True),
            {'penalty_type': 'elastic'}
        ),
        'FreeSurv': (
            FreeSurv(alpha=FreeSurv_lambda, verbose=True),
            {}
        )
    }

    final_results = {}
    top_Ns = [5, 10, 20, 30]
    
    # Print table header
    #print(f"{'Method':<12} | {'Set':<6} | {'Top N':<5} | {'C-index (Mean ± Std)':<25} | {'AUC (Mean ± Std)':<25}")
    #print("-" * 85)

    # 3. Main loop
    for method_name, (model, fit_kwargs) in selectors.items():
        # --- A. Feature Selection (done only on the training set) ---
        X_train_raw = datasets['train']['X_vals']
        Y_train_raw = datasets['train']['Y_vals']
        E_train_raw = datasets['train']['E_vals']
        max_val = Y_train_raw.max() # From original snippet, but max_val is not used here after filtering test sets.

        # Standardize features using StandardScaler
        # Fit on training data and transform all X_raw datasets
        scaler = StandardScaler()
        # X_train_raw is initially a numpy array, but later we reconstruct a DataFrame from it.
        # It's generally better to scale the X_vals (numpy array) that are passed to fit methods.
        X_train_scaled_for_selector = X_train_raw

        # Train the selector
        if method_name == 'FreeSurv':
            # Ensure Y_train_raw and E_train_raw are numpy arrays for FreeSurv.fit
            model.fit(X_train_scaled_for_selector, np.array(Y_train_raw), np.array(E_train_raw))
            # Assume FreeSurv has a feature_importances_ attribute
            importances = model.feature_importances_
        else:
            # L1 / ElasticNet
            model.fit(X_train_scaled_for_selector, Y_train_raw, E_train_raw, **fit_kwargs)
            importances = model.get_coefficients()

        # Get sorted indices (descending order by absolute importance)
        sorted_indices = np.argsort(np.abs(importances))[::-1]
        top_N_column = [feature_cols[fea] for fea in sorted_indices[:30]] # This line is from original code, might be for debugging/logging

        method_results = {'test1': {}, 'test2': {}}

        # --- B. Iterate through Top N values ---
        for top_n in top_Ns:
            # 1. Select column indices for the Top N features
            selected_idx = sorted_indices[:top_n]

            # 2. Helper function to construct subset DataFrame
            def get_subset_df(set_name, indices):
                # Apply the same scaler to test sets
                X_scaled = datasets[set_name]['X_vals'] #scaler.transform(datasets[set_name]['X_vals'])
                df_scaled = pd.DataFrame(X_scaled, columns=feature_cols) # Reconstruct DF with scaled features
                
                # Select feature columns by integer index from the scaled DataFrame
                df_feats = df_scaled.iloc[:, indices]

                # Select label columns
                df_labels = datasets[set_name]['df'][[duration_col, event_col]]
                return pd.concat([df_feats, df_labels], axis=1)

            df_train_sub = get_subset_df('train', selected_idx)

            # --- Evaluate on Test1 and Test2 ---
            for test_set_name in ['test1', 'test2']:
                df_test_sub = get_subset_df(test_set_name, selected_idx)

                # Initialize a new RF model for each evaluation
                rf_model = RFSurvival()

                # Bootstrap evaluation
                metrics = evaluate_with_bootstrap(
                    rf_model,
                    df_train_sub,
                    df_test_sub,
                    duration_col=duration_col,
                    event_col=event_col,
                    times=time_points,
                    n_bootstraps=1000,
                    seed=seed
                )

                # Store results
                method_results[test_set_name][top_n] = metrics

                # Print results (commented out the original print here, as main loop will print overall table)
                # print(f"{method_name:<12} | {test_set_name:<6} | {top_n:<5} | "
                #       f"{metrics['C-index_mean']:.4f} ± {metrics['C-index_std']:.4f}   | "
                #       f"{metrics['AUC_mean']:.4f} ± {metrics['AUC_std']:.4f}")

        final_results[method_name] = method_results
        # Print a separator after each method's results
     #   print("-" * 85)

    return final_results


def run_breast_cancer_experiments(data_name, time_points_auc, l1_lambda, elastic_lambda, FreeSurv_lambda, seed=42):
    if seed is not None: np.random.seed(seed)
    
    # data outcome
    if data_name=="BreastCancer":
        survival_outcomes = {
        'OS': {'duration_col': 'Survival.months', 'event_col': 'Survival.status'},
        'RFS': {'duration_col': 'RFS.months', 'event_col': 'RFS.status'}
        }
    elif data_name=="PDAC":
        survival_outcomes = {
        'OS': {'duration_col': 'Survival.months', 'event_col': 'Survival.status'},
        'RFS': {'duration_col': 'Recurr.months', 'event_col': 'Recurr.status'}
        }

    # Outcome -> Method -> Set -> Top N -> Metrics
    final_results = {} 
    top_Ns = [5, 10, 20, 30]
    datasets = {}
    # --- (OS, RFS) ---
    for outcome_name, outcome_cols in survival_outcomes.items():
        print(f"\n==================== Evaluating for Outcome: {outcome_name} ====================")
        duration_col = outcome_cols['duration_col']
        event_col = outcome_cols['event_col']     

        data_loader=Data(data_name = data_name, outcome=outcome_name, duration_col=duration_col, event_col=event_col) # Pass an instance of your Data class
        datasets[outcome_name] = {}
        # Load train, test1, test2
        for set_name in ["train", "test"]:
            Y, E, X = data_loader.get_data(data_name=data_name, set_name=set_name, duration_col=duration_col, event_col=event_col, outcome=outcome_name)


            # Convert to DataFrame and merge labels
            df = pd.DataFrame(X)
            # Record feature column names (assuming integer indices)
            feature_cols = df.columns.tolist()

            df[duration_col] = Y
            df[event_col] = E
            datasets[outcome_name][set_name] = {
                'df': df,
                'X_vals': X, # Original numerical matrix, for feature selector fit
                'Y_vals': Y,
                'E_vals': E
            }       
        # ========================================================


        selectors = {
            'L1': (
                CoxFeatureSelector(lambda1_l1=l1_lambda, verbose=True),
                {'penalty_type': 'L1'}
            ),
            'ElasticNet': (
                CoxFeatureSelector(alpha_elastic=elastic_lambda, l1_ratio_elastic=0.5, verbose=True),
                {'penalty_type': 'elastic'}
            ),
            'FreeSurv': (
                FreeSurv(alpha=FreeSurv_lambda, verbose=True),
                {}
            )
        }

        current_outcome_results = {}
        
        print(f"{'Method':<12} | {'Set':<6} | {'Top N':<5} | {'C-index (Mean ± Std)':<25} | {'AUC (Mean ± Std)':<25}")
        print("-" * 85)

        for method_name, (model, fit_kwargs) in selectors.items():
            # X_vals from processed_datasets are already scaled NumPy arrays
            X_train_for_selector = datasets[outcome_name]['train']['X_vals']
            Y_train_vals = datasets[outcome_name]['train']['Y_vals']
            E_train_vals = datasets[outcome_name]['train']['E_vals']

            scaler = StandardScaler()
            #X_train_for_selector = scaler.fit_transform(X_train_for_selector)

           
            # Note: StandardScaler is now applied inside load_and_preprocess_breast_cancer
            # So X_train_for_selector is already scaled. No need to re-scale here.


            # Train the selector
            if method_name == 'FreeSurv':
                model.fit(X_train_for_selector, np.array(Y_train_vals), np.array(E_train_vals))
                importances = model.feature_importances_
            else:
                model.fit(X_train_for_selector, Y_train_vals, E_train_vals, **fit_kwargs)
                importances = model.get_coefficients()
            
            # Get sorted indices (descending order by absolute importance)
            sorted_indices = np.argsort(np.abs(importances))[::-1]
            
            current_method_results = {'test':{}} # 存储当前方法在 test1/test2 上的结果

            for top_n in top_Ns:
                selected_idx = sorted_indices[:top_n]
                
                def get_subset_df_for_rf(set_name, outcome_name, indices):
                    df_orig = datasets[outcome_name][set_name]['df'] # This df is already scaled and has correct labels
                    
                    df_feats = df_orig.iloc[:, indices] # Select feature columns by integer index
                    df_labels = df_orig[[duration_col, event_col]]
                    return pd.concat([df_feats, df_labels], axis=1)

                df_train_sub = get_subset_df_for_rf('train', outcome_name, selected_idx)
                
                for test_set_name in ['test']:
                    df_test_sub = get_subset_df_for_rf(test_set_name, outcome_name, selected_idx)
                    
                    rf_model = RFSurvival() 
                    
                    metrics = evaluate_with_bootstrap(
                        rf_model, 
                        df_train_sub, 
                        df_test_sub,
                        duration_col=duration_col,
                        event_col=event_col,
                        times=time_points_auc,
                        n_bootstraps=1000, 
                        seed=seed
                    )
                    
                    current_method_results[test_set_name][top_n] = metrics
                    
                    print(f"{method_name:<12} | {test_set_name:<6} | {top_n:<5} | "
                          f"{metrics['C-index_mean']:.4f} ± {metrics['C-index_std']:.4f}   | "
                          f"{metrics['AUC_mean']:.4f} ± {metrics['AUC_std']:.4f}")

            current_outcome_results[method_name] = current_method_results
            print("-" * 85)
        
        final_results[outcome_name] = current_outcome_results

    return final_results





def print_results_table(results):
    """
    formalize the resutls table
    """
    all_top_ns = sorted(list(next(iter(results.values()))['test1'].keys()))
    test_sets = sorted(list(next(iter(results.values())).keys()))
    
    # 定义表格头部
    header = f"{'Method':<12} | {'Set':<6} | {'Metric':<10} | "
    for top_n in all_top_ns:
        header += f"{f'Top {top_n}':<24} | "
    print(header)
    print("-" * len(header))

    # 遍历每个方法
    for method_name, method_data in results.items():
        # 遍历每个测试集
        for set_name in test_sets:
            # 打印 C-index 均值和标准差
            c_index_row = f"{method_name:<12} | {set_name:<6} | {'C-index':<10} | "
            for top_n in all_top_ns:
                data = method_data[set_name].get(top_n, {})
                mean = data.get('C-index_mean', np.nan)
                std = data.get('C-index_std', np.nan)
                c_index_row += f"{mean:.4f} \u00B1 {std:.4f}{'':<13} | " # \u00B1 是 ± 符号
            print(c_index_row)

            # 打印 AUC 均值和标准差
            auc_row = f"{'':<12} | {'':<6} | {'AUC':<10} | "
            for top_n in all_top_ns:
                data = method_data[set_name].get(top_n, {})
                mean = data.get('AUC_mean', np.nan)
                std = data.get('AUC_std', np.nan)
                auc_row += f"{mean:.4f} \u00B1 {std:.4f}{'':<13} | "
            print(auc_row)
        print("-" * len(header))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated experiment settings")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--time_points_auc",
        type=int,
        default=36,
        help="time point to compute AUC"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="HCC",
        choices = ["HCC", "BreastCancer", "PDAC"],
        help="the dataset"
    )
    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=0.05,
        help="the sparsity penalty of Cox-Lasso (default: 0.1)"
    )

    parser.add_argument(
        "--elastic_lambda",
        type=float,
        default=0.1,
        help="the sparsity penalty of Cox-elastic (default: 0.1)"
    )

    parser.add_argument(
        "--FreeSurv_lambda",
        type=float,
        default=0.5,
        help="the sparsity penalty of pi-FreeSurv (default: 0.1)"
    )

    args = parser.parse_args()

    seed = args.seed
    dataset= args.dataset
    l1_lambda = args.l1_lambda
    time_points_auc = args.time_points_auc
    elastic_lambda = args.elastic_lambda
    FreeSurv_lambda = args.FreeSurv_lambda

    print("dataset =", dataset)
    print("time_points_auc =", time_points_auc)
    print("l1_lambda =", l1_lambda)
    print("elastic_lambda =", elastic_lambda)
    print("FreeSurv_lambda =", FreeSurv_lambda)
    print("seed =", seed)
    if dataset=="HCC":
        final_results = run_HCC_experiments(l1_lambda, time_points_auc, elastic_lambda, FreeSurv_lambda, seed=42)
        print_results_table(final_results)
    else:
        final_results = run_breast_cancer_experiments(dataset, time_points_auc, l1_lambda, elastic_lambda, FreeSurv_lambda, seed=42)
