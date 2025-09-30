"""
The module contains the set of helper function used for model saving and loading in different numbers/specifications.
"""
import os
import pickle
import joblib
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import average_precision_score
import time

def save_single_model(model, model_name, X_test, y_test, output_dir="saved_models", 
                     feature_names=None, model_tier=None, additional_info=None):
    """
    Save a single trained model with evaluation metrics
    
    Args:
        model: Trained model object
        model_name: Name for the model
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save model
        feature_names: List of feature names
        model_tier: Model performance tier (e.g., "Tier 1: Ultra-fast")
        additional_info: Additional metadata dictionary
        
    Returns:
        str: Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Measure inference latency
    latency_info = measure_model_latency(model, X_test)
    
    # Calculate performance metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    performance = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_mcc': matthews_corrcoef(y_test, y_pred),
        'test_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Save model
    joblib.dump(model, filepath)
    
    # Create metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'timestamp': timestamp,
        'filepath': filepath,
        'model_tier': model_tier,
        'feature_names': feature_names,
        'performance': performance,
        'latency': latency_info,
        'additional_info': additional_info or {}
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f"{model_name}_metadata_{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✓ Saved model: {filename}")
    print(f"  Performance: MCC={performance['test_mcc']:.3f}, AUC={performance['test_auc']:.3f}")
    print(f"  Latency: {latency_info['single_sample_ms']:.2f}ms")
    print(f"  Metadata: {os.path.basename(metadata_file)}")
    
    return filepath

def save_multiple_models(models_dict, X_test, y_test, output_dir="saved_models", 
                        feature_names=None):
    """
    Save multiple models at once
    
    Args:
        models_dict: Dictionary of {model_name: model_object}
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save models
        feature_names: List of feature names
        
    Returns:
        dict: Saved model information
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_models = {}
    
    print(f"Saving {len(models_dict)} models to: {output_dir}")
    print("-" * 50)
    
    for model_name, model in models_dict.items():
        try:
            # Determine model tier based on name
            model_tier = determine_model_tier(model_name)
            
            # Save individual model
            filepath = save_single_model(
                model=model,
                model_name=model_name,
                X_test=X_test,
                y_test=y_test,
                output_dir=output_dir,
                feature_names=feature_names,
                model_tier=model_tier
            )
            
            # Store info
            metadata_file = os.path.join(output_dir, f"{model_name}_metadata_{timestamp}.json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            saved_models[model_name] = metadata
            
        except Exception as e:
            print(f"✗ Failed to save {model_name}: {e}")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"models_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(saved_models, f, indent=2, default=str)
    
    print(f"\n✓ Successfully saved {len(saved_models)}/{len(models_dict)} models")
    print(f"✓ Summary: {summary_file}")
    
    return saved_models

def measure_model_latency(model, X_test, n_runs=100):
    """
    Measure model inference latency
    
    Args:
        model: Trained model
        X_test: Test data
        n_runs: Number of runs for averaging
        
    Returns:
        dict: Latency measurements
    """
    try:
        # Single sample latency
        single_sample = X_test.iloc[0:1] if hasattr(X_test, 'iloc') else X_test[0:1]
        
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model.predict(single_sample)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        # Batch latency
        batch_size = min(100, len(X_test))
        batch_sample = X_test.iloc[:batch_size] if hasattr(X_test, 'iloc') else X_test[:batch_size]
        
        batch_times = []
        for _ in range(10):
            start = time.perf_counter()
            model.predict(batch_sample)
            end = time.perf_counter()
            batch_times.append((end - start) * 1000)
        
        return {
            'single_sample_ms': np.mean(times),
            'single_sample_std': np.std(times),
            'batch_100_ms': np.mean(batch_times),
            'throughput_samples_per_sec': 1000 / np.mean(times) if np.mean(times) > 0 else 0
        }
        
    except Exception as e:
        print(f"Could not measure latency: {e}")
        return {
            'single_sample_ms': 0,
            'single_sample_std': 0,
            'batch_100_ms': 0,
            'throughput_samples_per_sec': 0
        }

def determine_model_tier(model_name):
    """
    Determine model tier based on model name
    
    Args:
        model_name: Name of the model
        
    Returns:
        str: Model tier classification
    """
    name_lower = model_name.lower()
    
    if 'logistic' in name_lower:
        return 'Tier 1: Ultra-fast (<1ms)'
    elif any(x in name_lower for x in ['lightgbm', 'lgb']) and 'fast' in name_lower:
        return 'Tier 2: Fast (1-5ms)'
    elif any(x in name_lower for x in ['xgb', 'xgboost']) and 'fast' in name_lower:
        return 'Tier 2: Fast (1-5ms)'
    elif any(x in name_lower for x in ['catboost', 'random_forest']) and 'fast' in name_lower:
        return 'Tier 3: Balanced (5-15ms)'
    elif 'accurate' in name_lower:
        return 'Tier 4: High accuracy (15-50ms)'
    elif any(x in name_lower for x in ['lightgbm', 'lgb', 'xgb', 'xgboost']):
        return 'Tier 3: Balanced (5-15ms)'
    else:
        return 'Unclassified'

def save_best_performing_model(models_dict, X_test, y_test, output_dir="saved_models", 
                              metric='mcc', model_name_prefix="best_model"):
    """
    Save the best performing model from a dictionary of models
    
    Args:
        models_dict: Dictionary of {model_name: model_object}
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save model
        metric: Metric to use for selection ('mcc', 'auc', 'f1')
        model_name_prefix: Prefix for saved model name
        
    Returns:
        str: Path to saved best model
    """
    print(f"Evaluating {len(models_dict)} models to find best by {metric.upper()}...")
    
    best_score = -1
    best_model_name = None
    best_model = None
    
    # Evaluate all models
    for model_name, model in models_dict.items():
        try:
            y_pred = model.predict(X_test)
            
            if metric == 'mcc':
                score = matthews_corrcoef(y_test, y_pred)
            elif metric == 'auc':
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                score = roc_auc_score(y_test, y_pred_proba)
            elif metric == 'prauc':
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                score = average_precision_score(y_test, y_pred_proba)
            elif metric == 'f1':
                score = f1_score(y_test, y_pred)
            else:
                score = accuracy_score(y_test, y_pred)
            
            print(f"  {model_name}: {metric.upper()}={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_model = model
                
        except Exception as e:
            print(f"  {model_name}: Error - {e}")
    
    if best_model is None:
        print("No valid models found")
        return None
    
    print(f"\nBest model: {best_model_name} ({metric.upper()}={best_score:.4f})")
    
    # Save best model
    final_name = f"{model_name_prefix}_{best_model_name}"
    filepath = save_single_model(
        model=best_model,
        model_name=final_name,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        additional_info={
            'selection_metric': metric,
            'selection_score': best_score,
            'original_model_name': best_model_name
        }
    )
    
    return filepath

def load_saved_model(filepath):
    """
    Load a saved model
    
    Args:
        filepath: Path to saved model file
        
    Returns:
        tuple: (model, metadata)
    """
    try:
        # Load model
        model = joblib.load(filepath)
        
        # Try to load metadata
        metadata = None
        metadata_file = filepath.replace('.pkl', '_metadata_')
        
        # Find metadata file (it has timestamp)
        directory = os.path.dirname(filepath)
        filename_base = os.path.basename(filepath).replace('.pkl', '')
        
        for file in os.listdir(directory):
            if file.startswith(filename_base) and 'metadata' in file and file.endswith('.json'):
                metadata_path = os.path.join(directory, file)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                break
        
        print(f"✓ Loaded model: {filepath}")
        if metadata:
            print(f"  Type: {metadata.get('model_type', 'Unknown')}")
            perf = metadata.get('performance', {})
            print(f"  Performance: MCC={perf.get('test_mcc', 0):.3f}")
        
        return model, metadata
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None, None

def quick_save_models(models_dict, X_test, y_test,path ="models"):
    """
    Quick save function - minimal setup required
    
    Args:
        models_dict: Dictionary of {model_name: model_object}
        X_test: Test features
        y_test: Test targets
        
    Returns:
        dict: Saved model information
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{path}\\models_{timestamp}"
    
    return save_multiple_models(models_dict, X_test, y_test, output_dir)