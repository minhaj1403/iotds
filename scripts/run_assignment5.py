#!/usr/bin/env python3
"""
Assignment 5: Ensemble methods

This script combines the best methods from previous assignments using ensemble techniques.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config.config import DATA_PATH, RANDOM_STATE, RESULTS_DIR
from src.data.loader import load_kemophone_data, set_seed
from src.features.extraction import reorder_and_split_features
from src.models.evxgb import EvXGBClassifier
from src.models.ensemble import evaluate_ensemble_cv, create_xgb_tabnet_ensemble
from src.utils.helpers import create_results_directory, save_experiment_config


def prepare_ensemble_data(X: pd.DataFrame) -> tuple:
    """
    Prepare data for ensemble models.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Tuple of (X_np, X_original, cat_cols, num_cols)
    """
    cat_cols = list(X.columns[X.dtypes == bool])
    num_cols = list(X.columns[~X.columns.isin(cat_cols)])
    
    # Create a copy for TabNet preprocessing
    X_tabnet = X.copy(deep=True)
    
    # Normalize numerical features for TabNet
    if num_cols:
        scaler = StandardScaler()
        X_tabnet[num_cols] = scaler.fit_transform(X_tabnet[num_cols])
    
    # Convert categorical boolean to float for TabNet
    if cat_cols:
        X_tabnet[cat_cols] = X_tabnet[cat_cols].astype(float)
    
    # Convert to numpy for TabNet
    X_np = X_tabnet.to_numpy()
    
    return X_np, X, cat_cols, num_cols


def test_ensemble_configurations(X_np: np.ndarray, X_original: pd.DataFrame, 
                                y: np.ndarray, groups: np.ndarray) -> dict:
    """
    Test different ensemble configurations.
    
    Args:
        X_np: Numpy array for TabNet
        X_original: Original DataFrame for XGBoost
        y: Target labels
        groups: Group labels
        
    Returns:
        Dictionary with ensemble results
    """
    results = {}
    
    # Test different weight combinations
    weight_combinations = [
        ([0.5, 0.5], "Equal weights"),
        ([0.6, 0.4], "XGBoost favored"),
        ([0.4, 0.6], "TabNet favored"),
        ([0.7, 0.3], "XGBoost heavy"),
        ([0.3, 0.7], "TabNet heavy"),
        ([0.55, 0.45], "XGBoost slight favor")
    ]
    
    for weights, description in weight_combinations:
        print(f"\nTesting ensemble: {description} (weights: {weights})")
        
        # Create ensemble configuration
        model_configs, ensemble_weights = create_xgb_tabnet_ensemble(
            weights=weights
        )
        
        # Evaluate ensemble
        try:
            ensemble_results = evaluate_ensemble_cv(
                X_np, y, groups,
                model_configs=model_configs,
                weights=ensemble_weights,
                cv_splits=5
            )
            
            # Calculate mean metrics
            metrics_df = pd.DataFrame(ensemble_results)
            mean_metrics = metrics_df.mean().to_dict()
            
            results[f"Ensemble_{description.replace(' ', '_')}"] = {
                'fold_results': ensemble_results,
                'mean_metrics': mean_metrics,
                'weights': weights,
                'description': description
            }
            
            print(f"Mean AUC: {mean_metrics.get('AUC', 0):.4f}")
            
        except Exception as e:
            print(f"Error in ensemble {description}: {str(e)}")
            continue
    
    return results


def compare_with_individual_models(X_np: np.ndarray, X_original: pd.DataFrame,
                                 y: np.ndarray, groups: np.ndarray) -> dict:
    """
    Compare ensemble with individual models.
    
    Args:
        X_np: Numpy array for TabNet
        X_original: Original DataFrame for XGBoost  
        y: Target labels
        groups: Group labels
        
    Returns:
        Dictionary with individual model results
    """
    results = {}
    
    # 1. XGBoost alone
    print("\nEvaluating XGBoost individual model...")
    from src.evaluation.cross_validation import perform_cross_validation, calculate_mean_metrics
    
    cat_cols = X_original.columns[X_original.dtypes == bool]
    
    xgb_estimator = EvXGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        eval_size=0.2,
        early_stopping_rounds=10,
        objective='binary:logistic',
        verbosity=0,
        learning_rate=0.01,
    )
    
    xgb_cv_results = perform_cross_validation(
        X_original, y, groups,
        estimator=xgb_estimator,
        cats=cat_cols,
        normalize=True,
        select=None,
        oversample=True,
        random_state=RANDOM_STATE
    )
    
    xgb_metrics = calculate_mean_metrics(xgb_cv_results)
    
    results['XGBoost_Individual'] = {
        'cv_results': xgb_cv_results,
        'mean_metrics': xgb_metrics,
        'model_type': 'XGBoost'
    }
    
    print(f"XGBoost Individual - Mean AUC: {xgb_metrics.get('AUC', 0):.4f}")
    
    # 2. TabNet alone
    print("\nEvaluating TabNet individual model...")
    from src.models.tabnet_model import TabNetModel
    
    tabnet_model = TabNetModel()
    tabnet_results = tabnet_model.fit_and_evaluate(X_np, y, groups, cv_splits=5)
    tabnet_metrics = tabnet_model.get_mean_metrics(tabnet_results)
    
    results['TabNet_Individual'] = {
        'fold_results': tabnet_results,
        'mean_metrics': tabnet_metrics,
        'model_type': 'TabNet'
    }
    
    print(f"TabNet Individual - Mean AUC: {tabnet_metrics.get('AUC', 0):.4f}")
    
    return results


def main():
    """Run Assignment 5: Ensemble methods."""
    print("="*60)
    print("ASSIGNMENT 5: ENSEMBLE METHODS")
    print("="*60)
    
    # Set random seed
    set_seed(RANDOM_STATE)
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create results directory
    results_dir = create_results_directory(os.path.join(RESULTS_DIR, "assignment5"))
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    print("\nLoading K-EmoPhone dataset...")
    X_raw, y, groups, t, datetimes = load_kemophone_data(DATA_PATH)
    
    # Extract and categorize features
    print("Extracting and categorizing features...")
    features = reorder_and_split_features(X_raw, y, groups, datetimes)
    
    # Use the best feature set (you can modify this based on previous assignments)
    # For now, we'll use all features
    X_selected = features["X_cleaned"]
    print(f"Using feature set with shape: {X_selected.shape}")
    
    # Prepare data for ensemble
    print("Preparing data for ensemble models...")
    X_np, X_original, cat_cols, num_cols = prepare_ensemble_data(X_selected)
    print(f"Categorical features: {len(cat_cols)}")
    print(f"Numerical features: {len(num_cols)}")
    
    # Test ensemble configurations
    print("\n" + "="*50)
    print("TESTING ENSEMBLE CONFIGURATIONS")
    print("="*50)
    
    ensemble_results = test_ensemble_configurations(
        X_np, X_original, features["y"], features["groups"]
    )
    
    # Compare with individual models
    print("\n" + "="*50)
    print("COMPARING WITH INDIVIDUAL MODELS")
    print("="*50)
    
    individual_results = compare_with_individual_models(
        X_np, X_original, features["y"], features["groups"]
    )
    
    # Combine all results
    all_results = {**ensemble_results, **individual_results}
    
    # Print comprehensive comparison
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    comparison_data = []
    for model_name, result in all_results.items():
        auc = result['mean_metrics'].get('AUC', 0)
        f1 = result['mean_metrics'].get('F1', 0)
        accuracy = result['mean_metrics'].get('Accuracy', 0)
        
        model_type = "Ensemble" if "Ensemble" in model_name else "Individual"
        
        row = {
            'Model_Name': model_name,
            'Model_Type': model_type,
            'Mean_AUC': auc,
            'Mean_F1': f1,
            'Mean_Accuracy': accuracy
        }
        
        if 'weights' in result:
            row['Ensemble_Weights'] = str(result['weights'])
        else:
            row['Ensemble_Weights'] = 'N/A'
            
        comparison_data.append(row)
        print(f"{model_name:<35} Type: {model_type:<10} AUC: {auc:.4f}")
    
    # Save results
    comparison_df = pd.DataFrame(comparison_data).sort_values('Mean_AUC', ascending=False)
    comparison_file = os.path.join(results_dir, "ensemble_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    # Save detailed results for each configuration
    for model_name, result in all_results.items():
        if 'fold_results' in result:
            # Ensemble or TabNet results
            fold_data = result['fold_results']
            if isinstance(fold_data, list) and len(fold_data) > 0:
                if isinstance(fold_data[0], dict):
                    fold_df = pd.DataFrame(fold_data)
                else:
                    # Handle other formats
                    fold_df = pd.DataFrame([{'Result': str(fold_data)}])
            else:
                fold_df = pd.DataFrame([{'Note': 'No fold results available'}])
        else:
            # XGBoost results
            fold_data = []
            for fold_result in result.get('cv_results', []):
                row = {'Fold': fold_result.name, 'Duration': fold_result.duration}
                row.update(fold_result.metrics)
                fold_data.append(row)
            fold_df = pd.DataFrame(fold_data)
        
        model_file = os.path.join(results_dir, f"{model_name}_detailed_results.csv")
        fold_df.to_csv(model_file, index=False)
    
    # Find best performing model
    best_model = comparison_df.iloc[0]
    print(f"\n" + "="*60)
    print("BEST PERFORMING MODEL")
    print("="*60)
    print(f"Model: {best_model['Model_Name']}")
    print(f"Type: {best_model['Model_Type']}")
    print(f"AUC: {best_model['Mean_AUC']:.4f}")
    if best_model['Ensemble_Weights'] != 'N/A':
        print(f"Ensemble Weights: {best_model['Ensemble_Weights']}")
    
    # Save experiment configuration
    config = {
        'ensemble_configurations_tested': len(ensemble_results),
        'individual_models_tested': len(individual_results),
        'best_model': best_model['Model_Name'],
        'best_auc': float(best_model['Mean_AUC']),
        'best_weights': best_model['Ensemble_Weights'] if best_model['Ensemble_Weights'] != 'N/A' else None,
        'feature_shape': X_selected.shape,
        'cuda_available': torch.cuda.is_available()
    }
    save_experiment_config(config, os.path.join(results_dir, "assignment5_config.json"))
    
    print(f"\nResults saved to: {results_dir}")
    print("Assignment 5 completed successfully!")
    
    # Summary of improvements
    if len(individual_results) >= 2:
        individual_aucs = [r['mean_metrics'].get('AUC', 0) for r in individual_results.values()]
        best_individual_auc = max(individual_aucs)
        best_ensemble_auc = best_model['Mean_AUC'] if best_model['Model_Type'] == 'Ensemble' else 0
        
        if best_ensemble_auc > 0:
            improvement = best_ensemble_auc - best_individual_auc
            print(f"\nEnsemble improvement over best individual model: {improvement:.4f} ({improvement/best_individual_auc*100:.2f}%)")


if __name__ == "__main__":
    main()
