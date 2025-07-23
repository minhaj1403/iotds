"""
Feature extraction and categorization utilities for K-EmoPhone dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict


def reorder_and_split_features(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, 
                               datetimes: np.ndarray) -> Dict[str, pd.DataFrame]:
    """
    Reorder data chronologically and extract all categorized features.

    Parameters:
        X: Raw features pickle loaded from the dataset
        y: Target labels
        groups: User IDs
        datetimes: Datetime array

    Returns:
        Dictionary containing all feature categories and metadata
    """
    # Reorder data chronologically
    df = pd.DataFrame({'user_id': groups, 'datetime': datetimes, 'label': y})
    df_merged = pd.merge(df, X, left_index=True, right_index=True)
    df_merged = df_merged.sort_values(by=['user_id', 'datetime'])

    groups = df_merged['user_id'].to_numpy()
    datetimes = df_merged['datetime'].to_numpy()
    y = df_merged['label'].to_numpy()
    X_cleaned = df_merged.drop(columns=['user_id', 'datetime', 'label'])

    # Categorized features
    feat_current = X_cleaned.loc[:, [('#VAL' in str(x)) or ('ESM#LastLabel' in str(x)) for x in X_cleaned.columns]]
    feat_dsc = X_cleaned.loc[:, [('#DSC' in str(x)) for x in X_cleaned.columns]]
    feat_yesterday = X_cleaned.loc[:, [('Yesterday' in str(x)) for x in X_cleaned.columns]]
    feat_today = X_cleaned.loc[:, [('Today' in str(x)) for x in X_cleaned.columns]]
    feat_ImmediatePast = X_cleaned.loc[:, [('ImmediatePast_15' in str(x)) for x in X_cleaned.columns]]

    # Fine-grained subcategories
    feat_current_sensor = X_cleaned.loc[:, [('#VAL' in str(x)) for x in X_cleaned.columns]]
    feat_current_ESM = X_cleaned.loc[:, [('ESM#LastLabel' in str(x)) for x in X_cleaned.columns]]
    feat_ImmediatePast_sensor = feat_ImmediatePast.loc[:, [('ESM' not in str(x)) for x in feat_ImmediatePast.columns]]
    feat_ImmediatePast_ESM = feat_ImmediatePast.loc[:, [('ESM' in str(x)) for x in feat_ImmediatePast.columns]]
    feat_today_sensor = feat_today.loc[:, [('ESM' not in str(x)) for x in feat_today.columns]]
    feat_today_ESM = feat_today.loc[:, [('ESM' in str(x)) for x in feat_today.columns]]
    feat_yesterday_sensor = feat_yesterday.loc[:, [('ESM' not in str(x)) for x in feat_yesterday.columns]]
    feat_yesterday_ESM = feat_yesterday.loc[:, [('ESM' in str(x)) for x in feat_yesterday.columns]]

    feat_sleep = X_cleaned.loc[:, [('Sleep' in str(x)) for x in X_cleaned.columns]]
    feat_time = X_cleaned.loc[:, [('Time' in str(x)) for x in X_cleaned.columns]]
    feat_pif = X_cleaned.loc[:, [('PIF' in str(x)) for x in X_cleaned.columns]]

    # Baseline feature combination
    feat_baseline = pd.concat([feat_time, feat_dsc, feat_current_sensor, feat_ImmediatePast_sensor], axis=1)

    return {
        "X_cleaned": X_cleaned,
        "y": y,
        "groups": groups,
        "datetimes": datetimes,
        "feat_current": feat_current,
        "feat_dsc": feat_dsc,
        "feat_yesterday": feat_yesterday,
        "feat_today": feat_today,
        "feat_ImmediatePast": feat_ImmediatePast,
        "feat_current_sensor": feat_current_sensor,
        "feat_current_ESM": feat_current_ESM,
        "feat_ImmediatePast_sensor": feat_ImmediatePast_sensor,
        "feat_ImmediatePast_ESM": feat_ImmediatePast_ESM,
        "feat_today_sensor": feat_today_sensor,
        "feat_today_ESM": feat_today_ESM,
        "feat_yesterday_sensor": feat_yesterday_sensor,
        "feat_yesterday_ESM": feat_yesterday_ESM,
        "feat_sleep": feat_sleep,
        "feat_time": feat_time,
        "feat_pif": feat_pif,
        "feat_baseline": feat_baseline
    }


def get_feature_combinations() -> Dict[str, callable]:
    """
    Define feature combination strategies for experimentation.
    
    Returns:
        Dictionary mapping combination names to functions that create feature sets
    """
    def create_feature_combinations(features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create different feature combinations for experiments."""
        return {
            "baseline": features["feat_baseline"],
            
            "all_sensors": features["X_cleaned"],
            
            "baseline+today+current_esm+pid+sleep": features["feat_baseline"]
                .join(features["feat_today_sensor"])
                .join(features["feat_today_ESM"])
                .join(features["feat_current_ESM"])
                .join(features["feat_pif"])
                .join(features["feat_sleep"]),
            
            "baseline+today+current_esm+sleep": features["feat_baseline"]
                .join(features["feat_today_sensor"])
                .join(features["feat_today_ESM"])
                .join(features["feat_current_ESM"])
                .join(features["feat_sleep"]),
            
            "baseline+today+current_esm+yesterday+immediate_past_esm": features["feat_baseline"]
                .join(features["feat_today_sensor"])
                .join(features["feat_today_ESM"])
                .join(features["feat_current_ESM"])
                .join(features["feat_yesterday_sensor"])
                .join(features["feat_yesterday_ESM"])
                .join(features["feat_ImmediatePast_ESM"]),
            
            "baseline+today+current_esm+yesterday+pid+sleep": features["feat_baseline"]
                .join(features["feat_today_sensor"])
                .join(features["feat_today_ESM"])
                .join(features["feat_current_ESM"])
                .join(features["feat_yesterday_sensor"])
                .join(features["feat_yesterday_ESM"])
                .join(features["feat_pif"])
                .join(features["feat_sleep"]),
            
            "baseline+today+current_esm+yesterday-no_immediate_past": features["feat_baseline"]
                .join(features["feat_today_sensor"])
                .join(features["feat_today_ESM"])
                .join(features["feat_current_ESM"])
                .join(features["feat_yesterday_sensor"])
                .join(features["feat_yesterday_ESM"])
                .drop(columns=features["feat_ImmediatePast_sensor"].columns),
            
            "baseline+today+current_esm+yesterday": features["feat_baseline"]
                .join(features["feat_today_sensor"])
                .join(features["feat_today_ESM"])
                .join(features["feat_current_ESM"])
                .join(features["feat_yesterday_sensor"])
                .join(features["feat_yesterday_ESM"]),
            
            "baseline+today+current_esm": features["feat_baseline"]
                .join(features["feat_today_sensor"])
                .join(features["feat_today_ESM"])
                .join(features["feat_current_ESM"]),
            
            "current+ImmediatePast": features["feat_current_sensor"]
                .join(features["feat_ImmediatePast_sensor"]),
            
            "current": features["feat_current"],
            
            "dsc": features["feat_dsc"],
            
            "sensor+time": features["feat_current_sensor"].join(features["feat_time"]),
        }
    
    return create_feature_combinations


def get_feature_summary(features: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """
    Get a summary of feature counts for each category.
    
    Args:
        features: Dictionary of feature DataFrames
        
    Returns:
        Dictionary mapping feature category names to feature counts
    """
    feature_groups = {
        "feat_time": features.get("feat_time"),
        "feat_dsc": features.get("feat_dsc"),
        "feat_current_sensor": features.get("feat_current_sensor"),
        "feat_current_ESM": features.get("feat_current_ESM"),
        "feat_ImmediatePast_sensor": features.get("feat_ImmediatePast_sensor"),
        "feat_ImmediatePast_ESM": features.get("feat_ImmediatePast_ESM"),
        "feat_today_sensor": features.get("feat_today_sensor"),
        "feat_today_ESM": features.get("feat_today_ESM"),
        "feat_yesterday_sensor": features.get("feat_yesterday_sensor"),
        "feat_yesterday_ESM": features.get("feat_yesterday_ESM"),
        "feat_sleep": features.get("feat_sleep"),
        "feat_pif": features.get("feat_pif"),
    }

    return {name: data.shape[1] if data is not None else 0 for name, data in feature_groups.items()}
