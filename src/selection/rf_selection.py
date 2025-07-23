"""
Random Forest-based feature selection utilities.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Dict


def get_rf_feature_ranking(X: pd.DataFrame, y: np.ndarray, 
                          max_samples: int = 1000, random_state: int = 42) -> pd.Series:
    """
    Train a Random Forest classifier and rank features by their importance scores.

    Parameters:
        X (pd.DataFrame): Input feature matrix.
        y (array-like): Target labels.
        max_samples (int): Unused here but included for API consistency.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.Series: Features ranked by importance (descending order) based on Random Forest.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    return pd.Series(importances, index=X.columns).sort_values(ascending=False)


def select_top_features_by_rf(X: pd.DataFrame, y: np.ndarray, 
                             top_n: int = 20) -> pd.DataFrame:
    """
    Select the top-N most important features using Random Forest importance scores.

    Parameters:
        X (pd.DataFrame): Input feature matrix.
        y (array-like): Target labels.
        top_n (int): Number of top features to select.

    Returns:
        pd.DataFrame: Reduced feature matrix with top-N Random Forest-ranked features.
    """
    ranking = get_rf_feature_ranking(X, y)
    top_features = ranking.head(top_n).index
    return X[top_features]


def evaluate_rf_selection(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                         estimator, TOP_N_list: List[int] = None,
                         feature_sets: Dict[str, pd.DataFrame] = None) -> Dict[str, Dict]:
    """
    Evaluate Random Forest-based feature selection across different top-N values.
    
    Args:
        X, y, groups: Dataset
        estimator: Model to evaluate
        TOP_N_list: List of top-N values to test
        feature_sets: Dictionary of feature sets to test
        
    Returns:
        Dictionary containing results for each configuration
    """
    from src.evaluation.cross_validation import perform_cross_validation, calculate_mean_metrics
    
    if TOP_N_list is None:
        TOP_N_list = [10, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300]
    
    if feature_sets is None:
        feature_sets = {"all_features": X}
    
    results = {}
    
    for TOP_N in TOP_N_list:
        for fs_name, X_base in feature_sets.items():
            
            # Skip if not enough features
            if X_base.shape[1] < TOP_N:
                continue

            print(f"\nFeature Set: {fs_name}, TOP_N: {TOP_N}")

            # Get all training data across folds using StratifiedGroupKFold
            # This avoids data leakage by only using training data for feature selection
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            train_indices = []

            for train_idx, _ in cv.split(X_base, y, groups):
                train_indices.extend(train_idx)

            # Remove duplicates from aggregated training indices
            unique_train_indices = sorted(set(train_indices))

            # Extract merged training data for RF feature selection
            X_train_merged = X_base.iloc[unique_train_indices]
            y_train_merged = y[unique_train_indices]

            # Run RF feature selection on merged training data
            X_selected = select_top_features_by_rf(X_train_merged, y_train_merged, top_n=TOP_N)
            selected_features = X_selected.columns.tolist()
            selected_features.sort()

            print(f"Selected {len(selected_features)} features from merged training data.")

            # Apply selected features to full dataset for evaluation
            X_rf = X_base[selected_features]

            # Identify categorical columns
            cat_cols = X_rf.columns[X_rf.dtypes == bool]

            # Run model evaluation using existing CV pipeline
            cv_results = perform_cross_validation(
                X_rf, y, groups,
                estimator=estimator,
                cats=cat_cols,
                normalize=True,
                select=None,  # RF already selects
                oversample=True,
                random_state=42
            )

            # Calculate mean metrics
            mean_metrics = calculate_mean_metrics(cv_results)
            
            config_name = f"{fs_name}_top{TOP_N}"
            results[config_name] = {
                'mean_metrics': mean_metrics,
                'cv_results': cv_results,
                'selected_features': selected_features,
                'config': {'feature_set': fs_name, 'top_n': TOP_N}
            }

            print(f"Mean AUC: {mean_metrics.get('AUC', 'N/A'):.4f}")

    return results


def save_rf_results(results: Dict[str, Dict], output_dir: str) -> None:
    """
    Save Random Forest evaluation results to files.
    
    Args:
        results: Results from evaluate_rf_selection
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary_data = []
    for config_name, result in results.items():
        row = {'Configuration': config_name}
        row.update(result['mean_metrics'])
        row.update(result['config'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'rf_summary.csv'), index=False)
    
    # Save individual results and features
    for config_name, result in results.items():
        # Save CV results
        cv_data = []
        for fold_result in result['cv_results']:
            row = {'Fold': fold_result.name, 'Duration': fold_result.duration}
            row.update(fold_result.metrics)
            cv_data.append(row)
        
        cv_df = pd.DataFrame(cv_data)
        cv_filename = f"rf_{config_name}.csv"
        cv_df.to_csv(os.path.join(output_dir, cv_filename), index=False)
        
        # Save selected features
        features_filename = f"rf_{config_name}_features.csv"
        pd.Series(result['selected_features']).to_csv(
            os.path.join(output_dir, features_filename),
            index=False, header=False
        )


def get_best_rf_config(results: Dict[str, Dict], metric: str = 'AUC') -> tuple:
    """
    Find the best Random Forest configuration based on a specific metric.
    
    Args:
        results: Results from evaluate_rf_selection
        metric: Metric to optimize
        
    Returns:
        Tuple of (best_config_name, best_score, best_features)
    """
    best_score = -float('inf')
    best_config_name = None
    best_features = None
    
    for config_name, result in results.items():
        score = result['mean_metrics'].get(metric, -float('inf'))
        if score > best_score:
            best_score = score
            best_config_name = config_name
            best_features = result['selected_features']
    
    return best_config_name, best_score, best_features
