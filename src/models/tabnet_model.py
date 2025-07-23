"""
TabNet implementation for tabular data classification.
"""

import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Dict, Any


class TabNetModel:
    """
    Wrapper class for TabNet classifier with cross-validation support.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize TabNet model with given parameters.
        
        Args:
            **kwargs: Parameters for TabNetClassifier
        """
        self.params = kwargs
        self.default_params = {
            'device_name': 'cuda',
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'lambda_sparse': 1e-4,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2),
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'scheduler_params': dict(gamma=0.95, step_size=20),
            'mask_type': 'sparsemax',
            'verbose': 0,
            'seed': 42
        }
        
        # Merge default and provided parameters
        self.model_params = {**self.default_params, **self.params}
    
    def create_model(self) -> TabNetClassifier:
        """Create a new TabNet model instance."""
        return TabNetClassifier(**self.model_params)
    
    def fit_and_evaluate(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                        cv_splits: int = 5) -> List[Dict[str, float]]:
        """
        Perform cross-validation training and evaluation.
        
        Args:
            X: Feature matrix
            y: Target labels
            groups: Group labels for cross-validation
            cv_splits: Number of cross-validation splits
            
        Returns:
            List of dictionaries containing metrics for each fold
        """
        cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            print(f"\nFold {fold + 1}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create and train model
            model = self.create_model()
            model.fit(
                X_train=X_train, 
                y_train=y_train,
                eval_set=[(X_test, y_test)],
                eval_name=["val"],
                eval_metric=["auc"],
                max_epochs=200,
                patience=20,
                batch_size=1024,
                virtual_batch_size=128,
            )
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            
            fold_results.append({
                'Accuracy': acc,
                'F1': f1,
                'AUC': auc,
                'Fold': fold + 1
            })
        
        return fold_results
    
    def get_mean_metrics(self, fold_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate mean metrics across all folds.
        
        Args:
            fold_results: Results from cross-validation
            
        Returns:
            Dictionary of mean metrics
        """
        metrics_df = pd.DataFrame(fold_results)
        return metrics_df.mean().to_dict()


def create_tabnet_with_hyperparams(hyperparams: Dict[str, Any]) -> TabNetModel:
    """
    Create TabNet model with specific hyperparameters.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        Configured TabNetModel instance
    """
    return TabNetModel(**hyperparams)
