"""
Enhanced XGBoost Classifier with built-in validation set approach for early stopping.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from typing import Union


class EvXGBClassifier(BaseEstimator):
    """
    Enhanced XGBClassifier with built-in validation set approach for early stopping.
    """
    
    def __init__(
        self,
        eval_size=None,
        eval_metric='logloss',
        early_stopping_rounds=10,
        random_state=None,
        **kwargs
    ):
        """
        Initializes the custom XGBoost Classifier.

        Args:
            eval_size (float): The proportion of the dataset to include in the evaluation split.
            eval_metric (str): The evaluation metric used for model training.
            early_stopping_rounds (int): The number of rounds to stop training if hold-out metric doesn't improve.
            random_state (int): Seed for the random number generator for reproducibility.
            **kwargs: Additional arguments to be passed to the underlying XGBClassifier.
        """
        self.random_state = random_state
        self.eval_size = eval_size
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        
        # Initialize the XGBClassifier with specified arguments and GPU acceleration.
        self.model = XGBClassifier(
            random_state=self.random_state,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            tree_method="hist", 
            device="cuda",  # Use GPU for acceleration
            **kwargs
        )

    @property
    def feature_importances_(self):
        """Returns the feature importances from the fitted model."""
        return self.model.feature_importances_

    @property
    def feature_names_in_(self):
        """Returns the feature names from the input dataset used for fitting."""
        return self.model.feature_names_in_

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray):
        """
        Fit the XGBoost model with optional early stopping using a validation set.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Training features.
            y (np.ndarray): Target values.
        """
        if self.eval_size:
            # Split data for early stopping evaluation if eval_size is specified.
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                X, y, test_size=self.eval_size, random_state=self.random_state
            )
            # Fit the model with early stopping.
            self.model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            # Fit the model without early stopping.
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X, y, verbose=False)

        # Store the best iteration number for predictions.
        booster = self.model.get_booster()
        self.best_iteration_ = (
            booster.best_iteration if hasattr(booster, "best_iteration") else None
        )
        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict the classes for the given features.

        Args:
            X (pd.DataFrame): Input features.
        """
        if self.best_iteration_ is not None:
            return self.model.predict(X, iteration_range=(0, self.best_iteration_ + 1))
        else:
            return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict the class probabilities for the given features.

        Args:
            X (pd.DataFrame): Input features.
        """
        if self.best_iteration_ is not None:
            return self.model.predict_proba(X, iteration_range=(0, self.best_iteration_ + 1))
        else:
            return self.model.predict_proba(X)
