#!/usr/bin/env python3
"""
Assignment 1: Feature combination experiments

This script tests different combinations of features to improve model performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config.config import DATA_PATH, RANDOM_STATE, RESULTS_DIR
from src.data.loader import load_kemophone_data, set_seed
from src.features.extraction import reorder_and_split_features, get_feature_combinations
from src.models.evxgb import EvXGBClassifier
from src.evaluation.cross_validation import perform_cross_validation, calculate_mean_metrics, save_cv_results
from src.utils.helpers import create_results_directory, print_results_summary


def main():
    """Run Assignment 1: Feature combination experiments."""
    print("="*60)
    print("ASSIGNMENT 1: FEATURE COMBINATION EXPERIMENTS")
    print("="*60)
    
    # Set random seed
    set_seed(RANDOM_STATE)
    
    # Create results directory
    results_dir = create_results_directory(os.path.join(RESULTS_DIR, "assignment1"))
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    print("\nLoading K-EmoPhone dataset...")
    X_raw, y, groups, t, datetimes = load_kemophone_data(DATA_PATH)
    
    # Extract and categorize features
    print("Extracting and categorizing features...")
    features = reorder_and_split_features(X_raw, y, groups, datetimes)
    
    # Get feature combinations
    create_combinations = get_feature_combinations()
    feature_sets = create_combinations(features)
    
    print(f"Testing {len(feature_sets)} feature combinations...")
    
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
    
    # Store results
    results = {}
    
    # Test each feature combination
    for name, X_feat in feature_sets.items():
        print(f"\n--- Testing: {name} ---")
        print(f"Feature shape: {X_feat.shape}")
        
        # Get categorical columns
        cat_cols = X_feat.columns[X_feat.dtypes == bool]
        print(f"Categorical features: {len(cat_cols)}")
        
        # Run cross-validation
        cv_results = perform_cross_validation(
            X_feat, features["y"], features["groups"],
            estimator=estimator,
            cats=cat_cols,
            normalize=True,
            select=None,  # No feature selection for Assignment 1
            oversample=True,
            random_state=RANDOM_STATE
        )
        
        # Calculate metrics
        mean_metrics = calculate_mean_metrics(cv_results)
        mean_auc = mean_metrics.get('AUC', 0)
        
        print(f"Mean AUC: {mean_auc:.4f}")
        
        # Store results
        results[name] = {
            'mean_metrics': mean_metrics,
            'cv_results': cv_results,
            'feature_shape': X_feat.shape,
            'n_categorical': len(cat_cols)
        }
        
        # Save individual results
        output_file = os.path.join(results_dir, f"assignment1_{name}_cv_results.csv")
        save_cv_results(cv_results, output_file)
    
    # Print summary
    print_results_summary(results, metric='AUC')
    
    # Save overall summary
    summary_data = []
    for name, result in results.items():
        row = {
            'Feature_Combination': name,
            'Mean_AUC': result['mean_metrics'].get('AUC', 0),
            'Feature_Count': result['feature_shape'][1],
            'Sample_Count': result['feature_shape'][0],
            'Categorical_Features': result['n_categorical']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data).sort_values('Mean_AUC', ascending=False)
    summary_file = os.path.join(results_dir, "assignment1_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nSummary saved to: {summary_file}")
    print("Assignment 1 completed successfully!")


if __name__ == "__main__":
    main()
