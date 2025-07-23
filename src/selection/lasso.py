"""
LASSO-based feature selection utilities.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from typing import List, Dict


def create_lasso_selectors(C_values: List[float] = None, 
                          thresholds: List = None) -> List[SelectFromModel]:
    """
    Create multiple LASSO selectors with different configurations.
    
    Args:
        C_values: List of regularization strengths to try
        thresholds: List of thresholds for feature selection
        
    Returns:
        List of SelectFromModel selectors
    """
    if C_values is None:
        C_values = [0.1, 1.0, 10.0]
    
    if thresholds is None:
        thresholds = [0.001, 0.005, 'mean']
    
    selectors = []
    for c in C_values:
        for thresh in thresholds:
            selector = SelectFromModel(
                estimator=LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=c,
                    random_state=42,
                    max_iter=4000
                ),
                threshold=thresh
            )
            selectors.append(selector)
    
    return selectors


def get_lasso_selector_config(C: float = 1.0, threshold: float = 0.005) -> SelectFromModel:
    """
    Create a single LASSO selector with specified parameters.
    
    Args:
        C: Regularization strength
        threshold: Feature selection threshold
        
    Returns:
        Configured SelectFromModel selector
    """
    return SelectFromModel(
        estimator=LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=C,
            random_state=42,
            max_iter=4000
        ),
        threshold=threshold
    )


def evaluate_lasso_configurations(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                                 estimator, feature_sets: Dict[str, pd.DataFrame],
                                 C_values: List[float] = None, 
                                 thresholds: List = None) -> Dict[str, Dict]:
    """
    Evaluate different LASSO configurations across multiple feature sets.
    
    Args:
        X, y, groups: Dataset
        estimator: Model to evaluate
        feature_sets: Dictionary of feature sets to test
        C_values: Regularization strengths to test
        thresholds: Thresholds to test
        
    Returns:
        Dictionary containing results for each configuration
    """
    from src.evaluation.cross_validation import perform_cross_validation, calculate_mean_metrics
    
    if C_values is None:
        C_values = [0.1, 1.0, 10.0]
    
    if thresholds is None:
        thresholds = [0.001, 0.005, 'mean']
    
    results = {}
    
    for fs_name, X_feat in feature_sets.items():
        cat_cols = X_feat.columns[X_feat.dtypes == bool]
        
        for c in C_values:
            for thresh in thresholds:
                config_name = f"{fs_name}_C={c}_thresh={thresh}"
                print(f"\nEvaluating: {config_name}")
                
                selector = get_lasso_selector_config(C=c, threshold=thresh)
                
                cv_results = perform_cross_validation(
                    X=X_feat,
                    y=y,
                    groups=groups,
                    estimator=estimator,
                    cats=cat_cols,
                    normalize=True,
                    select=[selector],
                    oversample=True,
                    random_state=42
                )
                
                mean_metrics = calculate_mean_metrics(cv_results)
                results[config_name] = {
                    'mean_metrics': mean_metrics,
                    'cv_results': cv_results,
                    'config': {'C': c, 'threshold': thresh, 'feature_set': fs_name}
                }
                
                print(f"Mean AUC: {mean_metrics.get('AUC', 'N/A'):.4f}")
    
    return results


def get_best_lasso_config(results: Dict[str, Dict], metric: str = 'AUC') -> tuple:
    """
    Find the best LASSO configuration based on a specific metric.
    
    Args:
        results: Results from evaluate_lasso_configurations
        metric: Metric to optimize
        
    Returns:
        Tuple of (best_config_name, best_score, best_config)
    """
    best_score = -float('inf')
    best_config_name = None
    best_config = None
    
    for config_name, result in results.items():
        score = result['mean_metrics'].get(metric, -float('inf'))
        if score > best_score:
            best_score = score
            best_config_name = config_name
            best_config = result['config']
    
    return best_config_name, best_score, best_config


def save_lasso_results(results: Dict[str, Dict], output_dir: str) -> None:
    """
    Save LASSO evaluation results to files.
    
    Args:
        results: Results from evaluate_lasso_configurations
        output_dir: Directory to save results
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary_data = []
    for config_name, result in results.items():
        row = {'Configuration': config_name}
        row.update(result['mean_metrics'])
        row.update(result['config'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'lasso_summary.csv'), index=False)
    
    # Save individual results
    for config_name, result in results.items():
        cv_data = []
        for fold_result in result['cv_results']:
            row = {'Fold': fold_result.name, 'Duration': fold_result.duration}
            row.update(fold_result.metrics)
            cv_data.append(row)
        
        cv_df = pd.DataFrame(cv_data)
        filename = f"lasso_{config_name.replace('=', '_').replace('.', '_')}.csv"
        cv_df.to_csv(os.path.join(output_dir, filename), index=False)
