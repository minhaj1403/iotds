"""
Ensemble methods for combining multiple models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Dict, Tuple, Any

from src.models.evxgb import EvXGBClassifier
from src.models.tabnet_model import TabNetModel


class SoftVotingEnsemble:
    """
    Soft voting ensemble that combines predictions from multiple models.
    """
    
    def __init__(self, models: List[Any], weights: List[float] = None):
        """
        Initialize the ensemble.
        
        Args:
            models: List of model instances
            weights: List of weights for each model (must sum to 1)
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit all models in the ensemble.
        
        Args:
            X: Training features
            y: Training labels
        """
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get weighted average of predicted probabilities.
        
        Args:
            X: Test features
            
        Returns:
            Weighted average probabilities
        """
        predictions = []
        for model in self.models:
            proba = model.predict_proba(X)
            if proba.ndim == 2:
                proba = proba[:, 1]  # Get positive class probabilities
            predictions.append(proba)
        
        # Weighted average
        ensemble_proba = np.average(predictions, weights=self.weights, axis=0)
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Get ensemble predictions.
        
        Args:
            X: Test features
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def evaluate_ensemble_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        model_configs: List[Dict], weights: List[float],
                        cv_splits: int = 5) -> List[Dict[str, float]]:
    """
    Evaluate ensemble using cross-validation.
    
    Args:
        X: Feature matrix
        y: Target labels
        groups: Group labels for CV
        model_configs: List of model configurations
        weights: Weights for ensemble
        cv_splits: Number of CV splits
        
    Returns:
        List of fold results
    """
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\nFold {fold + 1}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create and train models
        models = []
        for config in model_configs:
            if config['type'] == 'xgboost':
                model = EvXGBClassifier(**config['params'])
            elif config['type'] == 'tabnet':
                model = TabNetModel(**config['params']).create_model()
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            model.fit(X_train, y_train)
            models.append(model)
        
        # Create ensemble and make predictions
        ensemble = SoftVotingEnsemble(models, weights)
        
        # Get predictions
        ensemble_proba = ensemble.predict_proba(X_test)
        ensemble_preds = ensemble.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, ensemble_preds)
        f1 = f1_score(y_test, ensemble_preds)
        auc = roc_auc_score(y_test, ensemble_proba)
        
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        fold_results.append({
            'Fold': fold + 1,
            'Accuracy': acc,
            'F1': f1,
            'AUC': auc
        })
    
    return fold_results


def create_xgb_tabnet_ensemble(xgb_params: Dict = None, tabnet_params: Dict = None,
                              weights: List[float] = None) -> Tuple[List[Dict], List[float]]:
    """
    Create configuration for XGBoost + TabNet ensemble.
    
    Args:
        xgb_params: XGBoost parameters
        tabnet_params: TabNet parameters
        weights: Model weights
        
    Returns:
        Tuple of (model_configs, weights)
    """
    if xgb_params is None:
        xgb_params = {
            'random_state': 42,
            'eval_metric': 'logloss',
            'eval_size': 0.2,
            'early_stopping_rounds': 10,
            'objective': 'binary:logistic',
            'verbosity': 0,
            'learning_rate': 0.01
        }
    
    if tabnet_params is None:
        tabnet_params = {
            'device_name': 'cuda',
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'lambda_sparse': 1e-4,
            'verbose': 0,
            'seed': 42
        }
    
    if weights is None:
        weights = [0.55, 0.45]  # Slightly favor XGBoost
    
    model_configs = [
        {'type': 'xgboost', 'params': xgb_params},
        {'type': 'tabnet', 'params': tabnet_params}
    ]
    
    return model_configs, weights
