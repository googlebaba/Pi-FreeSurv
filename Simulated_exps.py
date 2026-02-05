from Baselines import CoxFeatureSelector
import argparse
import numpy as np
from TreeSurvival import RFSurvival
from simulated_data_generation import SurvivalDataGenerator
import pandas as pd
from tqdm import tqdm
from FreeSurv import FreeSurv
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_experiment_uncorrelated(n_features, n_samples, ind, top_n_select, mode, l1_lambda, elastic_lambda, FreeSurv_lambda, seed=None):
    results = {
        'L1': {'sensitivity': [], 'specificity': []},
        'ElasticNet': {'sensitivity': [], 'specificity': []},
        'FreeSurv': {'sensitivity': [], 'specificity': []}  # 新增
    }

    if seed is not None:
        np.random.seed(seed)

    n_runs = 10
    print(f"Starting experiment with {n_runs} runs...\n")

    for i in range(n_runs):
        current_seed = seed + i if seed is not None else None
        generator = SurvivalDataGenerator(seed=current_seed)

        Y_uncorr, E_uncorr, X_uncorr = generator.generate_uncorrelated_data(
            n_fea=n_features,
            n_samples=n_samples,
            mode=mode,
            ind=ind
        )

        # Ground Truth label
        true_relevant_features_idx = [n_features + idx for idx in [-4, -3, -2, -1]]
        y_true_feature_labels = np.zeros(n_features, dtype=int)
        y_true_feature_labels[true_relevant_features_idx] = 1

        # --- Model 1: L1 Penalty (Top N) ---
        cox_selector_l1 = CoxFeatureSelector(lambda1_l1=l1_lambda, verbose=True) # 设为 False 避免刷屏
        cox_selector_l1.fit(X_uncorr, Y_uncorr, E_uncorr, penalty_type='L1', top_n=top_n_select, y_true_feature_labels=y_true_feature_labels)
        metrics_l1 = cox_selector_l1.get_selection_metrics(y_true_feature_labels)

        if metrics_l1:
            results['L1']['sensitivity'].append(metrics_l1['sensitivity'])
            results['L1']['specificity'].append(metrics_l1['specificity'])

        # --- Model 2: ElasticNet Penalty (Top N) ---
        cox_selector_elastic = CoxFeatureSelector(alpha_elastic=elastic_lambda, l1_ratio_elastic=0.5, verbose=True)
        cox_selector_elastic.fit(X_uncorr, Y_uncorr, E_uncorr, penalty_type='elastic', top_n=top_n_select, y_true_feature_labels=y_true_feature_labels)
        metrics_elastic = cox_selector_elastic.get_selection_metrics(y_true_feature_labels)

        if metrics_elastic:
            results['ElasticNet']['sensitivity'].append(metrics_elastic['sensitivity'])
            results['ElasticNet']['specificity'].append(metrics_elastic['specificity'])

        # --- Model 3: FreeSurv (Top N) ---
        model_fs = FreeSurv(alpha=FreeSurv_lambda, verbose=True)
        model_fs.fit(X_uncorr, Y_uncorr, E_uncorr)

        metrics_fs = model_fs.evaluate_fs(y_true_feature_labels=y_true_feature_labels)
        results['FreeSurv']['sensitivity'].append(metrics_fs['sensitivity'])
        results['FreeSurv']['specificity'].append(metrics_fs['specificity'])

        print(f"Run {i+1}/{n_runs} finished.")

    print("\n" + "="*60)
    print(f"{'Model':<20} | {'Metric':<12} | {'Mean':<8} | {'Variance':<8} | {'Std Dev':<8}")
    print("-" * 60)

    for model_name, metrics in results.items():
        for metric_name in ['sensitivity', 'specificity']:
            values = metrics[metric_name]
            if len(values) > 0:
                mean_val = np.mean(values)
                var_val = np.var(values)
                std_val = np.std(values)
                print(f"{model_name:<20} | {metric_name.capitalize():<12} | {mean_val:.4f}   | {var_val:.4f}   | {std_val:.4f}")
            else:
                print(f"{model_name:<20} | {metric_name.capitalize():<12} | N/A        | N/A        | N/A")
        print("-" * 60)




def run_experiment_correlated(n_features, n_samples, ind, top_n_select, mode, l1_lambda, elastic_lambda, FreeSurv_lambda, seed=None):
    results = {
        'L1': {
            'sensitivity': [], 'specificity': [], 
            'c_index_iid': [], 'c_index_ood': []
        },
        'ElasticNet': {
            'sensitivity': [], 'specificity': [], 
            'c_index_iid': [], 'c_index_ood': []
        },
        'FreeSurv': {  # 新增 FreeSurv
            'sensitivity': [], 'specificity': [], 
            'c_index_iid': [], 'c_index_ood': []
        }
    }

    if seed is not None:
        np.random.seed(seed)
    
    n_runs = 10
    duration_col = 'duration'
    event_col = 'event'
    
    def make_dataframe(X, Y, E, duration_col, event_col):
        df = pd.DataFrame(X)
        df[duration_col] = Y.flatten()
        df[event_col] = E.flatten()
        return df

    def evaluate_with_rf(selected_idx, df_train, df_iid, df_ood, duration_col, event_col):
        if len(selected_idx) == 0:
            cols_to_use = [c for c in df_train.columns if c not in [duration_col, event_col]]
        else:
            cols_to_use = list(selected_idx)
            
        cols_all = cols_to_use + [duration_col, event_col]
        
        rf = RFSurvival(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=2)
        rf.fit(df_train[cols_all], duration_col=duration_col, event_col=event_col)
        
        c_iid = rf.score(df_iid[cols_all], duration_col=duration_col, event_col=event_col)
        c_ood = rf.score(df_ood[cols_all], duration_col=duration_col, event_col=event_col)
        return c_iid, c_ood

    print(f"Starting experiment with {n_runs} runs...\n")
    
    for i in range(n_runs):
        current_seed = seed + i if seed is not None else None
        generator = SurvivalDataGenerator(seed=current_seed)

        # --- Data Generation ---
        Y_uncorr_train, E_uncorr_train, X_uncorr_train = generator.generate_correlated_data(
            n_samples=n_samples, begin=0.6, end=0.9, mode=mode, ind=ind
        )
        Y_uncorr_iid, E_uncorr_iid, X_uncorr_iid = generator.generate_correlated_data(
            n_samples=n_samples, begin=0.6, end=0.9, mode=mode, ind=ind
        )
        Y_uncorr_ood, E_uncorr_ood, X_uncorr_ood = generator.generate_correlated_data(
            n_samples=n_samples, begin=0, end=0.5, mode=mode, ind=ind
        )

        df_train = make_dataframe(X_uncorr_train, Y_uncorr_train, E_uncorr_train, duration_col, event_col)
        df_iid = make_dataframe(X_uncorr_iid, Y_uncorr_iid, E_uncorr_iid, duration_col, event_col)
        df_ood = make_dataframe(X_uncorr_ood, Y_uncorr_ood, E_uncorr_ood, duration_col, event_col)

        # Ground Truth Label
        true_relevant_features_idx = [X_uncorr_train.shape[1] + k for k in [-4, -3, -2, -1]]
        y_true_feature_labels = np.zeros(X_uncorr_train.shape[1], dtype=int)                                  
        y_true_feature_labels[true_relevant_features_idx] = 1                                    

        # ==================== Model 1: L1 Penalty ====================
        cox_l1 = CoxFeatureSelector(lambda1_l1=l1_lambda, verbose=True)
        cox_l1.fit(X_uncorr_train, Y_uncorr_train, E_uncorr_train, penalty_type='L1', top_n=top_n_select, y_true_feature_labels=y_true_feature_labels)
        
        # Metrics
        metrics_l1 = cox_l1.get_selection_metrics(y_true_feature_labels)
        if metrics_l1:
            results['L1']['sensitivity'].append(metrics_l1['sensitivity'])
            results['L1']['specificity'].append(metrics_l1['specificity'])
        
        # RF Evaluation
        l1_idx = cox_l1.get_selected_feature_indices()
        c_iid_l1, c_ood_l1 = evaluate_with_rf(l1_idx, df_train, df_iid, df_ood, duration_col, event_col)
        results['L1']['c_index_iid'].append(c_iid_l1)
        results['L1']['c_index_ood'].append(c_ood_l1)

        # ==================== Model 2: ElasticNet Penalty ====================
        cox_el = CoxFeatureSelector(alpha_elastic=elastic_lambda, l1_ratio_elastic=0.5, verbose=True)
        cox_el.fit(X_uncorr_train, Y_uncorr_train, E_uncorr_train, penalty_type='elastic', top_n=top_n_select, y_true_feature_labels=y_true_feature_labels)
        
        # Metrics
        metrics_el = cox_el.get_selection_metrics(y_true_feature_labels)
        if metrics_el:
            results['ElasticNet']['sensitivity'].append(metrics_el['sensitivity'])
            results['ElasticNet']['specificity'].append(metrics_el['specificity'])

        # RF Evaluation
        el_idx = cox_el.get_selected_feature_indices()
        c_iid_el, c_ood_el = evaluate_with_rf(el_idx, df_train, df_iid, df_ood, duration_col, event_col)
        results['ElasticNet']['c_index_iid'].append(c_iid_el)
        results['ElasticNet']['c_index_ood'].append(c_ood_el)

        # ==================== Model 3: FreeSurv ====================
        model_fs = FreeSurv(alpha=FreeSurv_lambda, verbose=True)
        z_fs = model_fs.fit(X_uncorr_train, Y_uncorr_train, E_uncorr_train)
        
        # 1. Metrics (Sens/Spec)
        metrics_fs = model_fs.evaluate_fs(y_true_feature_labels=y_true_feature_labels)
        if metrics_fs:
            results['FreeSurv']['sensitivity'].append(metrics_fs['sensitivity'])
            results['FreeSurv']['specificity'].append(metrics_fs['specificity'])

        # 2. RF Evaluation
        fs_idx = model_fs.selected_indices_[:top_n_select]

        c_iid_fs, c_ood_fs = evaluate_with_rf(fs_idx, df_train, df_iid, df_ood, duration_col, event_col)
        results['FreeSurv']['c_index_iid'].append(c_iid_fs)
        results['FreeSurv']['c_index_ood'].append(c_ood_fs)

        print(f"Run {i+1}/{n_runs} finished.")

    print("\n" + "="*80)
    print(f"{'Model':<15} | {'Metric':<20} | {'Mean':<8} | {'Variance':<8} | {'Std Dev':<8}")
    print("-" * 80)

    for model_name, metrics in results.items():
        for metric_name in ['sensitivity', 'specificity', 'c_index_iid', 'c_index_ood']:
            values = metrics[metric_name]
            if len(values) > 0:
                mean_val = np.mean(values)
                var_val = np.var(values)
                std_val = np.std(values)
                print(f"{model_name:<15} | {metric_name:<20} | {mean_val:.4f}   | {var_val:.4f}   | {std_val:.4f}")
            else:
                 print(f"{model_name:<15} | {metric_name:<20} | N/A        | N/A        | N/A")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated experiment settings")
    
    parser.add_argument(
        "--exps_class",
        type=str,
        default="uncorrelated",
        choices=["correlated", "uncorrelated"],
        help='whether the features are correlated or uncorrelated.'
    )

    parser.add_argument(
        "--n_features",
        type=int,
        default=256,
        help="Number of features (default: 256)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)"
    )
    parser.add_argument(
        "--ind",
        type=bool,
        default=True,
        help="whether C is independent of Z"
    )
    parser.add_argument(
        "--top_n_select",
        type=int,
        default=4,
        help="Top-N features to select (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=0.1,
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
        default=0.1,
        help="the sparsity penalty of pi-FreeSurv (default: 0.1)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="Cox-additive",
        choices=["Cox-additive", "Cox-nonadditive", "log_T-additive", "log_T-nonadditive"],
        help='Data-generating mode (default: "Cox-nonadditive")'
    )
    args = parser.parse_args()

    exps_class = args.exps_class
    if exps_class == "correlated":
        n_features= 12
    else:
        n_features = args.n_features 
    n_samples = args.n_samples
    ind = args.ind
    top_n_select = args.top_n_select
    seed = args.seed
    mode = args.mode
    l1_lambda = args.l1_lambda
    elastic_lambda = args.elastic_lambda
    FreeSurv_lambda = args.FreeSurv_lambda
    print("exps_class", exps_class)
    print("n_features =", n_features)
    print("n_samples  =", n_samples)
    print("ind  =", ind)

    print("top_n_select =", top_n_select)

    print("mode =", mode)
    print("l1_lambda =", l1_lambda)
    print("elastic_lambda =", elastic_lambda)
    print("FreeSurv_lambda =", FreeSurv_lambda)
    print("seed =", seed)
    if exps_class == "uncorrelated":
        run_experiment_uncorrelated(n_features, n_samples, ind, top_n_select, mode, l1_lambda, elastic_lambda, FreeSurv_lambda, seed)
    else:
        run_experiment_correlated(12, n_samples, ind, top_n_select, mode, l1_lambda, elastic_lambda, FreeSurv_lambda, seed)
