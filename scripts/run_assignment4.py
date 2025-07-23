#!/usr/bin/env python3
"""
Assignment 4: Deep learning models for tabular data

This script tests TabNet and other deep learning models designed for tabular data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config.config import DATA_PATH, RANDOM_STATE, RESULTS_DIR
from src.data.loader import load_kemophone_data, set_seed
from src.features.extraction import reorder_and_split_features
from src.models.evxgb import EvXGBClassifier
from src.models.tabnet_model import TabNetModel, create_tabnet_with_hyperparams
from src.evaluation.cross_validation import perform_cross_validation, calculate_mean_metrics
from src.utils.helpers import create_results_directory, save_experiment_config


def prepare_data_for_tabnet(X: pd.DataFrame) -> tuple:
    """
    Prepare data for TabNet model.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Tuple of (X_np, cat_cols, num_cols)
    """
    cat_cols = list(X.columns[X.dtypes == bool])
    num_cols = list(X.columns[~X.columns.isin(cat_cols)])
    
    # Create a copy for preprocessing
    X_prep = X.copy(deep=True)
    
    # Normalize numerical features
    if num_cols:
        scaler = StandardScaler()
        X_prep[num_cols] = scaler.fit_transform(X_prep[num_cols])
    
    # Convert categorical boolean to float
    if cat_cols:
        X_prep[cat_cols] = X_prep[cat_cols].astype(float)
    
    # Convert to numpy
    X_np = X_prep.to_numpy()
    
    return X_np, cat_cols, num_cols


def compare_models(X_np: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  X_original: pd.DataFrame) -> dict:
    """
    Compare TabNet with XGBoost baseline.
    
    Args:
        X_np: Numpy array for TabNet
        y: Target labels
        groups: Group labels
        X_original: Original DataFrame for XGBoost
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    # 1. TabNet with default parameters
    print("\n1. Evaluating TabNet (default parameters)...")
    tabnet_default = TabNetModel()
    tabnet_results = tabnet_default.fit_and_evaluate(X_np, y, groups, cv_splits=5)
    tabnet_metrics = tabnet_default.get_mean_metrics(tabnet_results)
    
    results['TabNet_Default'] = {
        'fold_results': tabnet_results,
        'mean_metrics': tabnet_metrics,
        'model_type': 'TabNet',
        'config': 'Default parameters'
    }
    
    print(f"TabNet Default - Mean AUC: {tabnet_metrics.get('AUC', 0):.4f}")
    
    # 2. TabNet with tuned parameters (example configuration)
    print("\n2. Evaluating TabNet (tuned parameters)...")
    tuned_params = {
        'n_d': 64,
        'n_a': 64, 
        'n_steps': 5,
        'gamma': 1.5,
        'lambda_sparse': 1e-4,
        'optimizer_params': dict(lr=2e-2),
        'scheduler_params': dict(gamma=0.95, step_size=20),
    }
    
    tabnet_tuned = create_tabnet_with_hyperparams(tuned_params)
    tabnet_tuned_results = tabnet_tuned.fit_and_evaluate(X_np, y, groups, cv_splits=5)
    tabnet_tuned_metrics = tabnet_tuned.get_mean_metrics(tabnet_tuned_results)
    
    results['TabNet_Tuned'] = {
        'fold_results': tabnet_tuned_results,
        'mean_metrics': tabnet_tuned_metrics,
        'model_type': 'TabNet',
        'config': tuned_params
    }
    
    print(f"TabNet Tuned - Mean AUC: {tabnet_tuned_metrics.get('AUC', 0):.4f}")
    
    # 3. XGBoost baseline for comparison
    print("\n3. Evaluating XGBoost baseline...")
    cat_cols = X_original.columns[X_original.dtypes == bool]
    
    xgb_estimator = EvXGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        eval_size=0.2,
        early_stopping_rounds=10,
        objective='binary:logistic',
        verbosity=0,
        learning_rate=0.01,
    )
    
    xgb_cv_results = perform_cross_validation(
        X_original, y, groups,
        estimator=xgb_estimator,
        cats=cat_cols,
        normalize=True,
        select=None,
        oversample=True,
        random_state=RANDOM_STATE
    )
    
    xgb_metrics = calculate_mean_metrics(xgb_cv_results)
    
    results['XGBoost_Baseline'] = {
        'cv_results': xgb_cv_results,
        'mean_metrics': xgb_metrics,
        'model_type': 'XGBoost',
        'config': 'Baseline parameters'
    }
    
    print(f"XGBoost Baseline - Mean AUC: {xgb_metrics.get('AUC', 0):.4f}")
    
    return results


def main():
    """Run Assignment 4: Deep learning models for tabular data."""
    print("="*60)
    print("ASSIGNMENT 4: DEEP LEARNING MODELS FOR TABULAR DATA")
    print("="*60)
    
    # Set random seed
    set_seed(RANDOM_STATE)
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create results directory
    results_dir = create_results_directory(os.path.join(RESULTS_DIR, "assignment4"))
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    print("\nLoading K-EmoPhone dataset...")
    X_raw, y, groups, t, datetimes = load_kemophone_data(DATA_PATH)
    
    # Extract and categorize features
    print("Extracting and categorizing features...")
    features = reorder_and_split_features(X_raw, y, groups, datetimes)
    
    # For this assignment, we'll test different feature sets
    feature_sets_to_test = {
        "all_features": features["X_cleaned"],
        "baseline": features["feat_baseline"],
        # If you have SHAP top features from Assignment 2, you can load them here
        # "shap_top_60": load_shap_features_if_available()
    }
    
    all_results = {}
    
    for fs_name, X_feat in feature_sets_to_test.items():
        print(f"\n{'='*50}")
        print(f"TESTING FEATURE SET: {fs_name.upper()}")
        print(f"Feature shape: {X_feat.shape}")
        print("="*50)
        
        # Prepare data for TabNet
        X_np, cat_cols, num_cols = prepare_data_for_tabnet(X_feat)
        print(f"Categorical features: {len(cat_cols)}")
        print(f"Numerical features: {len(num_cols)}")
        
        # Compare models
        fs_results = compare_models(X_np, features["y"], features["groups"], X_feat)
        
        # Store results with feature set prefix
        for model_name, result in fs_results.items():
            all_results[f"{fs_name}_{model_name}"] = result
    
    # Print overall comparison
    print("\n" + "="*60)
    print("OVERALL MODEL COMPARISON")
    print("="*60)
    
    comparison_data = []
    for model_name, result in all_results.items():
        auc = result['mean_metrics'].get('AUC', 0)
        f1 = result['mean_metrics'].get('F1', 0)
        accuracy = result['mean_metrics'].get('Accuracy', 0)
        
        row = {
            'Model_Configuration': model_name,
            'Model_Type': result['model_type'],
            'Mean_AUC': auc,
            'Mean_F1': f1,
            'Mean_Accuracy': accuracy
        }
        comparison_data.append(row)
        print(f"{model_name:<30} AUC: {auc:.4f}")
    
    # Save results
    comparison_df = pd.DataFrame(comparison_data).sort_values('Mean_AUC', ascending=False)
    comparison_file = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    # Save detailed results for each model
    for model_name, result in all_results.items():
        if 'fold_results' in result:
            # TabNet results
            fold_data = result['fold_results']
            fold_df = pd.DataFrame(fold_data)
        else:
            # XGBoost results
            fold_data = []
            for fold_result in result['cv_results']:
                row = {'Fold': fold_result.name, 'Duration': fold_result.duration}
                row.update(fold_result.metrics)
                fold_data.append(row)
            fold_df = pd.DataFrame(fold_data)
        
        model_file = os.path.join(results_dir, f"{model_name}_detailed_results.csv")
        fold_df.to_csv(model_file, index=False)
    
    # Save experiment configuration
    config = {
        'feature_sets_tested': list(feature_sets_to_test.keys()),
        'models_evaluated': list(set([r['model_type'] for r in all_results.values()])),
        'cuda_available': torch.cuda.is_available(),
        'best_model': comparison_df.iloc[0]['Model_Configuration'],
        'best_auc': comparison_df.iloc[0]['Mean_AUC']
    }
    save_experiment_config(config, os.path.join(results_dir, "assignment4_config.json"))
    
    print(f"\nBest performing model: {config['best_model']}")
    print(f"Best AUC: {config['best_auc']:.4f}")
    print(f"\nResults saved to: {results_dir}")
    print("Assignment 4 completed successfully!")


if __name__ == "__main__":
    main()
