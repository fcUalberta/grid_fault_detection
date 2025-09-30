"""
The module contains the set of helper function used across the different stages in ML lifecycle
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import warnings
warnings.filterwarnings('ignore')


def prepare_data(feature_df, target_col='target', test_size=0.2, 
                     validation_size=0.2, stratify=True,random_state=123):
    """
    Prepare data for training and evaluation
    
    Args:
        feature_df: DataFrame with features and target
        target_col: Name of target column
        test_size: Proportion of data for testing
        validation_size: Proportion of training data for validation
        stratify: Whether to stratify splits
        
    Returns:
        dict: Dictionary containing train/val/test splits
    """
    print("Preparing data for model training...")
    
    # Separate features and target
    feature_cols = [col for col in feature_df.columns if col not in 
                   ['signal_id', 'chunk_id', target_col, 'id_measurement', 'phase']]
    
    X = feature_df[feature_cols]
    y = feature_df[target_col]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(y.value_counts(normalize=True))
    
    # First split: train+val vs test
    stratify_param = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=stratify_param
    )
    
    # Second split: train vs val
    stratify_param = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_size, random_state=random_state,
        stratify=stratify_param
    )
    
    data_splits = {
        'X_train': X_train,
        'X_val': X_val, 
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_cols
    }
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    return data_splits


def create_feature_tier_mapping(feature_names):
    """
    Create mapping of features to their computational tiers
    
    Args:
        feature_names: List of feature names
        
    Returns:
        dict: Feature name to tier mapping
    """
    tier_mapping = {}
    
    # Tier 1: Ultra-fast statistical features (<1ms)
    tier1_features = [
        'std', 'rms', 'mean_abs', 'min_val', 'max_val', 'peak_to_peak',
        'crest_factor', 'form_factor', 'mean_abs_deviation', 'variance'
    ]
    
    # Tier 2: Fast signal quality features (1-5ms)
    tier2_features = [
        'zero_crossing_rate', 'peak_count', 'peak_density', 'envelope_mean',
        'envelope_std', 'envelope_variation', 'high_freq_ratio'
    ]
    
    # Tier 3: Advanced features (5-20ms)
    tier3_features = [
        'spectral_centroid', 'dominant_frequency', 'dominant_magnitude',
        'dominant_power_ratio', 'autocorr_first_min'
    ]
    
    # Power system specific features
    power_system_features = [
        'thd_estimate', 'fundamental_frequency', 'transient_index',
        'max_gradient', 'amplitude_variation'
    ]
    
    # Wavelet features (Tier 3)
    wavelet_features = [f for f in feature_names if 'wavelet' in f]
    
    # Create mapping
    for feature in feature_names:
        # if any(s in feature for s in tier1_features):
        #     tier_mapping[feature] = 'Tier 1: Ultra-fast (<1ms)'
        if any(s in feature for s in tier2_features):
            tier_mapping[feature] = 'Tier 2: Fast'
        elif any(s in feature for s in tier3_features): 
        # or any(feature in s for s in wavelet_features):
            tier_mapping[feature] = 'Tier 3: Advanced'
        elif any(s in feature for s in power_system_features) :
            tier_mapping[feature] = 'Power System Specific'
        else:
            tier_mapping[feature] = 'Tier 1: Ultra-fast'
            
    return tier_mapping