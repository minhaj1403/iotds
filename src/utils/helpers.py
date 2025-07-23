"""
Utility functions and helpers for the K-EmoPhone stress prediction project.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List
from datetime import datetime


def create_results_directory(base_dir: str = "results") -> str:
    """
    Create a timestamped results directory.
    
    Args:
        base_dir: Base directory name
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_experiment_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save experiment configuration to a file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration
    """
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Recursively convert numpy types
    def clean_config(config):
        if isinstance(config, dict):
            return {k: clean_config(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [clean_config(v) for v in config]
        else:
            return convert_numpy(config)
    
    cleaned_config = clean_config(config)
    
    with open(output_path, 'w') as f:
        json.dump(cleaned_config, f, indent=2)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def print_results_summary(results: Dict[str, Dict], metric: str = 'AUC') -> None:
    """
    Print a summary of experimental results.
    
    Args:
        results: Dictionary of results from experiments
        metric: Metric to display in summary
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT RESULTS SUMMARY - {metric}")
    print(f"{'='*60}")
    
    # Sort by performance
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['mean_metrics'].get(metric, -float('inf')),
        reverse=True
    )
    
    print(f"{'Rank':<5} {'Configuration':<30} {metric:<10} {'Features':<10}")
    print("-" * 60)
    
    for rank, (config_name, result) in enumerate(sorted_results, 1):
        score = result['mean_metrics'].get(metric, 'N/A')
        n_features = result.get('config', {}).get('top_n', 'N/A')
        
        if isinstance(score, float):
            score_str = f"{score:.4f}"
        else:
            score_str = str(score)
            
        print(f"{rank:<5} {config_name[:29]:<30} {score_str:<10} {str(n_features):<10}")


def calculate_feature_stability(feature_lists: List[List[str]]) -> Dict[str, float]:
    """
    Calculate stability metrics for feature selection across folds.
    
    Args:
        feature_lists: List of feature lists from different folds
        
    Returns:
        Dictionary with stability metrics
    """
    if not feature_lists:
        return {}
    
    # Count feature occurrences
    all_features = set()
    for features in feature_lists:
        all_features.update(features)
    
    feature_counts = {}
    for feature in all_features:
        count = sum(1 for features in feature_lists if feature in features)
        feature_counts[feature] = count / len(feature_lists)
    
    # Calculate stability metrics
    n_folds = len(feature_lists)
    n_selected = np.mean([len(features) for features in feature_lists])
    
    # Jaccard stability (pairwise average)
    jaccard_scores = []
    for i in range(len(feature_lists)):
        for j in range(i + 1, len(feature_lists)):
            set_i = set(feature_lists[i])
            set_j = set(feature_lists[j])
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)
    
    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
    
    return {
        'n_folds': n_folds,
        'avg_features_selected': n_selected,
        'total_unique_features': len(all_features),
        'avg_jaccard_similarity': avg_jaccard,
        'feature_selection_frequencies': feature_counts
    }


def create_feature_comparison_report(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison report of different feature selection methods.
    
    Args:
        results: Results from different feature selection experiments
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for config_name, result in results.items():
        row = {
            'Configuration': config_name,
            'Method': 'Unknown',
            'Features_Used': 'Unknown'
        }
        
        # Extract method type
        if 'lasso' in config_name.lower():
            row['Method'] = 'LASSO'
        elif 'shap' in config_name.lower():
            row['Method'] = 'SHAP'
        elif 'rf' in config_name.lower():
            row['Method'] = 'Random Forest'
        
        # Add metrics
        row.update(result['mean_metrics'])
        
        # Add configuration details
        config = result.get('config', {})
        if 'top_n' in config:
            row['Features_Used'] = config['top_n']
        elif 'C' in config and 'threshold' in config:
            row['Features_Used'] = f"C={config['C']}, thresh={config['threshold']}"
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def validate_data_integrity(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> bool:
    """
    Validate the integrity of the dataset.
    
    Args:
        X: Feature matrix
        y: Target labels
        groups: Group labels
        
    Returns:
        True if data is valid, False otherwise
    """
    # Check shapes match
    if len(X) != len(y) or len(X) != len(groups):
        print(f"Shape mismatch: X={X.shape}, y={y.shape}, groups={groups.shape}")
        return False
    
    # Check for missing values in targets and groups
    if np.isnan(y).any():
        print("Missing values found in target labels")
        return False
    
    if pd.isna(groups).any():
        print("Missing values found in group labels")
        return False
    
    # Check feature data
    missing_features = X.columns[X.isnull().all()].tolist()
    if missing_features:
        print(f"Features with all missing values: {missing_features}")
        return False
    
    print("Data integrity check passed")
    return True
