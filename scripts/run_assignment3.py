#!/usr/bin/env python3
"""
Assignment 3: Hyperparameter tuning using Hyperopt

This script performs hyperparameter tuning for the XGBoost model using Hyperopt.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, hp, fmin, tpe
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, SMOTENC

from config.config import DATA_PATH, RANDOM_STATE, RESULTS_DIR, HYPEROPT_MAX_EVALS
from src.data.loader import load_kemophone_data, set_seed
from src.features.extraction import reorder_and_split_features
from src.models.evxgb import EvXGBClassifier
from src.evaluation.cross_validation import perform_cross_validation, calculate_mean_metrics
from src.utils.helpers import create_results_directory, save_experiment_config


def create_hyperopt_objective(X_selected, y, groups):
    """
    Create the objective function for hyperopt optimization.
    
    Args:
        X_selected: Selected features for tuning
        y: Target labels
        groups: Group labels for CV
        
    Returns:
        Objective function for hyperopt
    """
    # Define outer CV
    OUTER_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(params):
        val_scores = []
        
        # Outer loop: split into train_full / test (we will only use train_full for tuning)
        for train_full_idx, _ in OUTER_CV.split(X_selected, y, groups):
            X_train_full = X_selected.iloc[train_full_idx]
            y_train_full = y[train_full_idx]
            
            # Split 20% of the *training fold* into a validation set
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.20,
                stratify=y_train_full,
                random_state=42
            )
            
            # Normalize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # (Optional) Oversample on *training only*
            if np.any(X_train_scaled[:, -1] < 1):
                smote = SMOTENC(
                    categorical_features=[X_train_scaled.shape[1]-1],
                    random_state=int(params['random_state'])
                )
            else:
                smote = SMOTE(random_state=int(params['random_state']))
            X_train_os, y_train_os = smote.fit_resample(X_train_scaled, y_train)
            
            # Train & score on *validation only*
            clf = EvXGBClassifier(
                random_state=int(params['random_state']),
                eval_metric='logloss',
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                n_estimators=int(params['n_estimators']),
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
            )
            
            clf.fit(X_train_os, y_train_os)
            
            y_val_prob = clf.predict_proba(X_val_scaled)[:, 1]
            val_scores.append(roc_auc_score(y_val, y_val_prob))
        
        # Hyperopt minimizes "loss", so negate AUC
        return {'loss': -np.mean(val_scores), 'status': STATUS_OK}
    
    return objective


def main():
    """Run Assignment 3: Hyperparameter tuning."""
    print("="*60)
    print("ASSIGNMENT 3: HYPERPARAMETER TUNING WITH HYPEROPT")
    print("="*60)
    
    # Set random seed
    set_seed(RANDOM_STATE)
    
    # Create results directory
    results_dir = create_results_directory(os.path.join(RESULTS_DIR, "assignment3"))
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    print("\nLoading K-EmoPhone dataset...")
    X_raw, y, groups, t, datetimes = load_kemophone_data(DATA_PATH)
    
    # Extract and categorize features
    print("Extracting and categorizing features...")
    features = reorder_and_split_features(X_raw, y, groups, datetimes)
    
    # Use all features for hyperparameter tuning (or load best features from Assignment 2)
    print("Using all features for hyperparameter tuning...")
    X_selected = features["X_cleaned"]
    
    print(f"Feature matrix shape: {X_selected.shape}")
    
    # Define hyperparameter search space
    space = {
        'max_depth': hp.choice('max_depth', list(range(3, 10))),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'gamma': hp.uniform('gamma', 0, 5),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'random_state': 42
    }
    
    # Create objective function
    print("Creating hyperopt objective function...")
    objective = create_hyperopt_objective(X_selected, features["y"], features["groups"])
    
    # Run hyperopt
    print(f"Starting hyperparameter optimization with {HYPEROPT_MAX_EVALS} evaluations...")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=HYPEROPT_MAX_EVALS,
        trials=trials
    )
    
    # Process best parameters
    best_params = {k: v for k, v in best.items() if k != 'random_state'}
    best_params['max_depth'] = [3, 4, 5, 6, 7, 8, 9][best_params['max_depth']]
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['random_state'] = 42
    
    print("\nBest hyperparameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save best parameters
    best_params_file = os.path.join(results_dir, "best_hyperparameters.csv")
    pd.DataFrame([best_params]).to_csv(best_params_file, index=False)
    
    # Save experiment configuration
    config = {
        'search_space': {k: str(v) for k, v in space.items()},
        'max_evals': HYPEROPT_MAX_EVALS,
        'best_params': best_params,
        'best_score': -min([trial['result']['loss'] for trial in trials.trials]),
        'feature_shape': X_selected.shape
    }
    save_experiment_config(config, os.path.join(results_dir, "hyperopt_config.json"))
    
    # Evaluate tuned model with cross-validation
    print("\nEvaluating tuned model with cross-validation...")
    
    cat_cols = X_selected.columns[X_selected.dtypes == bool]
    
    # Create tuned estimator
    tuned_estimator = EvXGBClassifier(
        random_state=42,
        eval_metric='logloss',
        eval_size=0.2,
        early_stopping_rounds=10,
        objective='binary:logistic',
        verbosity=0,
        **{k: v for k, v in best_params.items() if k != 'random_state'}
    )
    
    # Perform cross-validation
    cv_results = perform_cross_validation(
        X_selected, features["y"], features["groups"],
        estimator=tuned_estimator,
        cats=cat_cols,
        normalize=True,
        select=None,
        oversample=True,
        random_state=42
    )
    
    # Calculate metrics
    mean_metrics = calculate_mean_metrics(cv_results)
    tuned_auc = mean_metrics.get('AUC', 0)
    
    print(f"\nTuned model performance:")
    print(f"Mean AUC: {tuned_auc:.4f}")
    
    # Compare with baseline (default parameters)
    print("\nComparing with baseline model (default parameters)...")
    baseline_estimator = EvXGBClassifier(
        random_state=42,
        eval_metric='logloss',
        eval_size=0.2,
        early_stopping_rounds=10,
        objective='binary:logistic',
        verbosity=0,
        learning_rate=0.01,
    )
    
    baseline_results = perform_cross_validation(
        X_selected, features["y"], features["groups"],
        estimator=baseline_estimator,
        cats=cat_cols,
        normalize=True,
        select=None,
        oversample=True,
        random_state=42
    )
    
    baseline_metrics = calculate_mean_metrics(baseline_results)
    baseline_auc = baseline_metrics.get('AUC', 0)
    
    print(f"Baseline model performance:")
    print(f"Mean AUC: {baseline_auc:.4f}")
    
    improvement = tuned_auc - baseline_auc
    print(f"\nImprovement: {improvement:.4f} ({improvement/baseline_auc*100:.2f}%)")
    
    # Save comparison results
    comparison_data = [
        {'Model': 'Baseline', 'Mean_AUC': baseline_auc},
        {'Model': 'Tuned', 'Mean_AUC': tuned_auc},
        {'Model': 'Improvement', 'Mean_AUC': improvement}
    ]
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = os.path.join(results_dir, "hyperopt_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"\nResults saved to: {results_dir}")
    print("Assignment 3 completed successfully!")


if __name__ == "__main__":
    main()
