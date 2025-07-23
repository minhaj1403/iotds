"""
Cross-validation utilities for model evaluation.
"""

import pandas as pd
import numpy as np
import time
import traceback
from dataclasses import dataclass
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, SMOTENC
from typing import List, Optional, Any


@dataclass
class FoldResult:
    """
    Data class to store results for each fold.
    """
    name: str
    metrics: dict
    duration: float


def log(message: str) -> None:
    """
    Simple logger function.
    
    Args:
        message: Message to log
    """
    print(message)


def train_fold(fold_name: str, X_train: pd.DataFrame, y_train: np.ndarray, 
               X_test: pd.DataFrame, y_test: np.ndarray, C_cat: np.ndarray, 
               C_num: np.ndarray, estimator: Any, normalize: bool, 
               select: Optional[List[SelectFromModel]], oversample: bool, 
               random_state: int) -> Optional[FoldResult]:
    """
    Trains and evaluates a model on a single fold.

    Args:
        fold_name: Name of the fold.
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        C_cat: List of categorical feature names.
        C_num: List of numerical feature names.
        estimator: Model to train.
        normalize: Whether to normalize numeric features.
        select: Feature selector(s).
        oversample: Whether to apply oversampling.
        random_state: Random seed.

    Returns:
        FoldResult object with metrics and duration.
    """
    try:
        start_time = time.time()

        # Normalize numeric features if requested
        if normalize:
            X_train_N = X_train[C_num].values
            X_test_N = X_test[C_num].values
            X_train_C = X_train[C_cat].values
            X_test_C = X_test[C_cat].values

            scaler = StandardScaler().fit(X_train_N)
            X_train_N = scaler.transform(X_train_N)
            X_test_N = scaler.transform(X_test_N)

            # Concatenate categorical and normalized numeric features
            X_train = pd.DataFrame(
                np.concatenate((X_train_C, X_train_N), axis=1),
                columns=np.concatenate((C_cat, C_num))
            )
            X_test = pd.DataFrame(
                np.concatenate((X_test_C, X_test_N), axis=1),
                columns=np.concatenate((C_cat, C_num))
            )

        # Feature selection if requested
        if select:
            if isinstance(select, SelectFromModel):
                select = [select]

            for s in select:
                support = s.fit(X_train.values, y_train).get_support()
                selected_cols = X_train.columns[support]
                C_cat = np.intersect1d(C_cat, selected_cols)
                C_num = np.intersect1d(C_num, selected_cols)

                X_train = X_train[selected_cols]
                X_test = X_test[selected_cols]

        # Oversampling if requested
        if oversample:
            if len(C_cat) > 0:
                sampler = SMOTENC(
                    categorical_features=[X_train.columns.get_loc(c) for c in C_cat],
                    random_state=random_state
                )
            else:
                sampler = SMOTE(random_state=random_state)
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        # Train the estimator and compute AUC
        estimator = clone(estimator).fit(X_train, y_train)
        y_pred = estimator.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)

        return FoldResult(
            name=fold_name,
            metrics={'AUC': auc_score},
            duration=time.time() - start_time
        )

    except Exception:
        log(f'Error in {fold_name}: {traceback.format_exc()}')
        return None


def perform_cross_validation(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, 
                           estimator: Any, cats: np.ndarray, normalize: bool = False, 
                           select: Optional[List[SelectFromModel]] = None, 
                           oversample: bool = False, random_state: Optional[int] = None,
                           cv_splits: int = 5) -> List[FoldResult]:
    """
    Performs cross-validation using StratifiedGroupKFold.

    Args:
        X: Feature DataFrame.
        y: Target array.
        groups: Group labels for the samples.
        estimator: Model to train.
        cats: List of categorical feature names.
        normalize: Whether to normalize numeric features.
        select: Feature selector(s).
        oversample: Whether to apply oversampling.
        random_state: Random seed.
        cv_splits: Number of cross-validation splits.

    Returns:
        List of FoldResult objects.
    """
    results = []

    # Use StratifiedGroupKFold for validation
    splitter = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Separate categorical and numeric columns
        C_cat = np.asarray(sorted(cats))
        C_num = np.asarray([col for col in X.columns if col not in C_cat])

        result = train_fold(f"Fold_{idx}", X_train, y_train, X_test, y_test,
                           C_cat, C_num, estimator, normalize, select, oversample, random_state)
        if result:
            results.append(result)

    # Print AUC for each fold
    for res in results:
        log(f"{res.name} - AUC: {res.metrics['AUC']:.4f} | Duration: {res.duration:.2f}s")

    return results


def calculate_mean_metrics(results: List[FoldResult]) -> dict:
    """
    Calculate mean metrics from cross-validation results.
    
    Args:
        results: List of FoldResult objects
        
    Returns:
        Dictionary with mean metrics
    """
    if not results:
        return {}
    
    # Extract all metrics
    all_metrics = {}
    for result in results:
        for metric_name, metric_value in result.metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(metric_value)
    
    # Calculate means
    mean_metrics = {
        metric_name: np.mean(values) 
        for metric_name, values in all_metrics.items()
    }
    
    return mean_metrics


def save_cv_results(results: List[FoldResult], filename: str) -> None:
    """
    Save cross-validation results to CSV file.
    
    Args:
        results: List of FoldResult objects
        filename: Output filename
    """
    # Convert results to DataFrame
    data = []
    for result in results:
        row = {'Fold': result.name, 'Duration': result.duration}
        row.update(result.metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add mean row
    mean_row = {'Fold': 'Mean', 'Duration': df['Duration'].mean()}
    for col in df.columns:
        if col not in ['Fold', 'Duration']:
            mean_row[col] = df[col].mean()
    
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    log(f"Results saved to {filename}")
