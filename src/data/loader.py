"""
Data loading and preprocessing utilities for K-EmoPhone dataset.
"""

import pickle
import pytz
from datetime import datetime
import pandas as pd
import numpy as np
import random
import torch
import os
from typing import Tuple

from config.config import RANDOM_STATE, DEFAULT_TZ


def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg: any) -> None:
    """
    Simple logging function with timestamp.
    
    Args:
        msg: Message to log
    """
    print('[{}] {}'.format(datetime.now().strftime('%y-%m-%d %H:%M:%S'), msg))


def load_kemophone_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the K-EmoPhone dataset from pickle file.
    
    Args:
        data_path (str): Path to the pickle file containing the dataset
        
    Returns:
        Tuple containing:
        - X: Feature DataFrame
        - y: Target labels array
        - groups: User IDs array
        - t: Time array
        - datetimes: Datetime array
    """
    try:
        with open(data_path, 'rb') as f:
            X, y, groups, t, datetimes = pickle.load(f)
        
        log(f"Successfully loaded K-EmoPhone data from {data_path}")
        log(f"Data shape: X={X.shape}, y={y.shape}, groups={groups.shape}")
        
        return X, y, groups, t, datetimes
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def preprocess_data(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, 
                   datetimes: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Basic preprocessing: sort data chronologically by user and datetime.
    
    Args:
        X: Feature DataFrame
        y: Target labels
        groups: User IDs
        datetimes: Datetime array
        
    Returns:
        Tuple of preprocessed data (X, y, groups, datetimes)
    """
    # Create DataFrame with metadata
    df = pd.DataFrame({
        'user_id': groups, 
        'datetime': datetimes, 
        'label': y
    })
    
    # Merge with features
    df_merged = pd.merge(df, X, left_index=True, right_index=True)
    
    # Sort by user and datetime
    df_merged = df_merged.sort_values(by=['user_id', 'datetime'])
    
    # Extract sorted data
    groups_sorted = df_merged['user_id'].to_numpy()
    datetimes_sorted = df_merged['datetime'].to_numpy()
    y_sorted = df_merged['label'].to_numpy()
    X_sorted = df_merged.drop(columns=['user_id', 'datetime', 'label'])
    
    log("Data preprocessed and sorted chronologically")
    
    return X_sorted, y_sorted, groups_sorted, datetimes_sorted
