#!/usr/bin/env python3
"""
Assignment 2: Feature selection experiments

This script tests different feature selection methods including LASSO, SHAP, and Random Forest.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config.config import DATA_PATH, RANDOM_STATE, RESULTS_DIR
from src.data.loader import load_kemophone_data, set_seed
from src.features.extraction import reorder_and_split_features
from src.models.evxgb import EvXGBClassifier
from src.selection.lasso import evaluate_lasso_configurations, save_lasso_results
from src.selection.shap_selection import evaluate_shap_selection, save_shap_results
from src.selection.rf_selection import evaluate_rf_selection, save_rf_results
from src.utils.helpers import create_results_directory, print_results_summary


def main():
    """Run Assignment 2: Feature selection experiments."""
    print("="*60)
    print("ASSIGNMENT 2: FEATURE SELECTION EXPERIMENTS")
    print("="*60)
    
    # Set random seed
    set_seed(RANDOM_STATE)
    
    # Create results directory
    results_dir = create_results_directory(os.path.join(RESULTS_DIR, "assignment2"))
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    print("\nLoading K-EmoPhone dataset...")
    X_raw, y, groups, t, datetimes = load_kemophone_data(DATA_PATH)
    
    # Extract and categorize features
    print("Extracting and categorizing features...")
    features = reorder_and_split_features(X_raw, y, groups, datetimes)
    
    # Initialize model
    estimator = EvXGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        eval_size=0.2,
        early_stopping_rounds=10,
        objective='binary:logistic',
        verbosity=0,
        learning_rate=0.01,
    )
    
    # Feature sets for LASSO experiments (high-performing combinations from Assignment 1)
    lasso_feature_sets = {
        "feat_baseline": features['feat_baseline'],
        "baseline+today+current_esm": features['feat_baseline']
            .join(features['feat_today_sensor'])
            .join(features['feat_today_ESM'])
            .join(features['feat_current_ESM']),
        "baseline+today+current_esm+yesterday": features['feat_baseline']
            .join(features['feat_today_sensor'])
            .join(features['feat_today_ESM'])
            .join(features['feat_current_ESM'])
            .join(features['feat_yesterday_sensor'])
            .join(features['feat_yesterday_ESM']),
        "baseline+today+current_esm+yesterday-no_immediate_past": features['feat_baseline']
            .join(features['feat_today_sensor'])
            .join(features['feat_today_ESM'])
            .join(features['feat_current_ESM'])
            .join(features['feat_yesterday_sensor'])
            .join(features['feat_yesterday_ESM'])
            .drop(columns=features['feat_ImmediatePast_sensor'].columns),
        "baseline+today+current_esm+yesterday+immediate_past_esm": features['feat_baseline']
            .join(features['feat_today_sensor'])
            .join(features['feat_today_ESM'])
            .join(features['feat_current_ESM'])
            .join(features['feat_yesterday_sensor'])
            .join(features['feat_yesterday_ESM'])
            .join(features['feat_ImmediatePast_ESM']),
    }
    
    # 1. LASSO-based feature selection
    print("\n" + "="*50)
    print("1. LASSO-BASED FEATURE SELECTION")
    print("="*50)
    
    lasso_results = evaluate_lasso_configurations(
        X=features["X_cleaned"],
        y=features["y"],
        groups=features["groups"],
        estimator=estimator,
        feature_sets=lasso_feature_sets,
        C_values=[0.1, 1.0, 10.0],
        thresholds=[0.001, 0.005, 'mean']
    )
    
    lasso_dir = os.path.join(results_dir, "lasso")
    save_lasso_results(lasso_results, lasso_dir)
    print_results_summary(lasso_results, metric='AUC')
    
    # 2. SHAP-based feature selection
    print("\n" + "="*50)
    print("2. SHAP-BASED FEATURE SELECTION")
    print("="*50)
    
    shap_feature_sets = {"all_features": features["X_cleaned"]}
    TOP_N_list = [10, 20, 30, 40, 45, 50, 60, 65, 70, 80, 90, 100, 200, 300, 500]
    
    shap_results = evaluate_shap_selection(
        X=features["X_cleaned"],
        y=features["y"],
        groups=features["groups"],
        estimator=estimator,
        TOP_N_list=TOP_N_list,
        feature_sets=shap_feature_sets
    )
    
    shap_dir = os.path.join(results_dir, "shap")
    save_shap_results(shap_results, shap_dir)
    print_results_summary(shap_results, metric='AUC')
    
    # 3. Random Forest-based feature selection
    print("\n" + "="*50)
    print("3. RANDOM FOREST-BASED FEATURE SELECTION")
    print("="*50)
    
    rf_feature_sets = {"all_features": features["X_cleaned"]}
    rf_TOP_N_list = [10, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300]
    
    rf_results = evaluate_rf_selection(
        X=features["X_cleaned"],
        y=features["y"],
        groups=features["groups"],
        estimator=estimator,
        TOP_N_list=rf_TOP_N_list,
        feature_sets=rf_feature_sets
    )
    
    rf_dir = os.path.join(results_dir, "rf")
    save_rf_results(rf_results, rf_dir)
    print_results_summary(rf_results, metric='AUC')
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL FEATURE SELECTION COMPARISON")
    print("="*60)
    
    all_results = {**lasso_results, **shap_results, **rf_results}
    print_results_summary(all_results, metric='AUC')
    
    # Save combined summary
    from src.utils.helpers import create_feature_comparison_report
    comparison_df = create_feature_comparison_report(all_results)
    comparison_file = os.path.join(results_dir, "feature_selection_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"\nComplete comparison saved to: {comparison_file}")
    print("Assignment 2 completed successfully!")


if __name__ == "__main__":
    main()
