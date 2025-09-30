"""
The module contains the set of helper function used for model evaluations and their visualizations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, matthews_corrcoef,
                           precision_score, recall_score, f1_score, accuracy_score,average_precision_score)
import xgboost as xgb
# import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import warnings


warnings.filterwarnings('ignore')

def measure_latency(func, *args, n_runs=100, **kwargs):
    """
    Measure execution latency of a function
    
    Args:
        func: Function to measure
        args: Function arguments
        n_runs: Number of runs for averaging
        kwargs: Function keyword arguments
        
    Returns:
        dict: Latency statistics
    """
    times = []
    memory_usage = []
    
    # Warm-up run
    func(*args, **kwargs)
    
    for _ in range(n_runs):
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        memory_usage.append(memory_after - memory_before)
    
    return {
        'mean_latency_ms': np.mean(times),
        'std_latency_ms': np.std(times),
        'min_latency_ms': np.min(times),
        'max_latency_ms': np.max(times),
        'p95_latency_ms': np.percentile(times, 95),
        'p99_latency_ms': np.percentile(times, 99),
        'mean_memory_mb': np.mean(memory_usage),
        'n_runs': n_runs
    }

def measure_training_latency(model, X_train, y_train):
    """
    Measure model training latency
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training targets
        
    Returns:
        dict: Training latency statistics
    """
    def train_func():
        model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        return model_copy.fit(X_train, y_train)
    
    return measure_latency(train_func, n_runs=5)  # Fewer runs for training

def measure_inference_latency(model, X_test, batch_sizes=[1, 10, 100, 1000]):
    """
    Measure model inference latency for different batch sizes
    
    Args:
        model: Trained model
        X_test: Test features
        batch_sizes: List of batch sizes to test
        
    Returns:
        dict: Inference latency statistics for different batch sizes
    """
    latency_results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(X_test):
            continue
            
        # Sample batch
        X_batch = X_test.iloc[:batch_size] if hasattr(X_test, 'iloc') else X_test[:batch_size]
        
        def predict_func():
            return model.predict(X_batch)
        
        def predict_proba_func():
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_batch)
            else:
                return model.predict(X_batch)
        
        # Measure prediction latency
        pred_latency = measure_latency(predict_func, n_runs=100)
        proba_latency = measure_latency(predict_proba_func, n_runs=100)
        
        latency_results[f'batch_{batch_size}'] = {
            'batch_size': batch_size,
            'predict_latency': pred_latency,
            'predict_proba_latency': proba_latency,
            'per_sample_predict_ms': pred_latency['mean_latency_ms'] / batch_size,
            'per_sample_proba_ms': proba_latency['mean_latency_ms'] / batch_size,
            'throughput_samples_per_sec': 1000 * batch_size / pred_latency['mean_latency_ms']
        }
    
    return latency_results


def plot_model_comparison_with_latency(results_df, figsize=(12, 12)):
    """
    Enhanced model comparison plot including latency metrics
    
    Args:
        results_df: DataFrame with evaluation results
        figsize: Figure size
    """
    if results_df.empty:
        print("No results to plot")
        return
        
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    # Sort for better visualization
    results_sorted = results_df.sort_values('test_mcc', ascending=True)
    
    metrics = [ 'test_mcc','test_pr_auc',  'single_latency_ms','test_auc', 'test_accuracy', 'test_f1', 'test_recall', 'throughput_per_sec']
    metric_labels = ['MCC', 'PR AUC','Latency (ms)','AUC', 'Accuracy', 'F1 Score', 'Recall',  'Throughput']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if i >= len(axes) or metric not in results_sorted.columns:
            continue
        # Sort by metric for better visualization
        sorted_data = results_df.sort_values(metric, ascending=True)    
        
        # Color bars based on grid suitability
        colors = ['yellowgreen' if suitable else 'maroon' for suitable in results_sorted['grid_suitable']]
        
        bars = axes[i].barh(range(len(results_sorted)), results_sorted[metric], color=colors, alpha=0.7)
        axes[i].set_yticks(range(len(results_sorted)))
        axes[i].set_yticklabels(results_sorted['model'], fontsize=8)
        axes[i].set_xlabel(label)
        axes[i].set_title(f'Model Comparison - {label}')
        axes[i].grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for j, (idx, row) in enumerate(results_sorted.iterrows()):
            value = row[metric]
            axes[i].text(value + 0.01 * (results_sorted[metric].max() - results_sorted[metric].min()), 
                       j, f'{value:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Comprehensive Model Performance Comparison\n(Green=Grid Suitable, Red=Not Suitable)', 
                 y=1.02, fontsize=12)
    plt.show()


def plot_latency_accuracy_tradeoff(evaluation_df, figsize=(12, 8)):
    """
    Plot latency vs accuracy trade-off for grid deployment
    
    Args:
        evaluation_df: DataFrame from evaluate_for_grid_deployment
        figsize: Figure size
    """
    if evaluation_df.empty:
        print("No evaluation results to plot")
        return
        
    plt.figure(figsize=figsize)
    
    # Color code by grid suitability
    colors = ['green' if suitable else 'red' for suitable in evaluation_df['grid_suitable']]
    sizes = [100 if suitable else 50 for suitable in evaluation_df['grid_suitable']]
    
    # Create scatter plot
    scatter = plt.scatter(evaluation_df['single_latency_ms'], evaluation_df['test_mcc'], 
                        c=colors, s=sizes, alpha=0.7, edgecolors='black')
    
    # Add model labels
    for _, row in evaluation_df.iterrows():
        plt.annotate(row['model'], 
                    (row['single_latency_ms'], row['test_mcc']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add requirement lines
    plt.axhline(y=0.3, color='blue', linestyle='--', alpha=0.7, label='Min MCC = 0.4')
    plt.axvline(x=10, color='blue', linestyle='--', alpha=0.7, label='Max Latency = 10ms')
    
    # Highlight grid-suitable region
    max_latency = evaluation_df['single_latency_ms'].max()
    max_mcc = evaluation_df['test_mcc'].max()
    
    if max_latency > 10:
        plt.axvspan(0, 10, ymin=0.3/max_mcc if max_mcc > 0 else 0, ymax=1, 
                   alpha=0.2, color='green', label='Grid Suitable Zone')
    
    plt.xlabel('Single Sample Latency (ms)')
    plt.ylabel('Test MCC Score')
    plt.title('Model Performance: Latency vs Accuracy Trade-off for Grid Deployment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add tier annotations
    if 'model_tier' in evaluation_df.columns:
        tier_info = evaluation_df.groupby('model_tier').agg({
            'single_latency_ms': 'mean',
            'test_mcc': 'mean'
        }).reset_index()
        
        for _, tier in tier_info.iterrows():
            if pd.notna(tier['single_latency_ms']) and pd.notna(tier['test_mcc']):
                plt.annotate(f"{tier['model_tier']}\n(avg)", 
                            (tier['single_latency_ms'], tier['test_mcc']),
                            xytext=(10, -10), textcoords='offset points',
                            fontsize=8, style='italic', alpha=0.7,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.5))
    
    # Add performance summary text
    suitable_count = evaluation_df['grid_suitable'].sum()
    total_count = len(evaluation_df)
    
    plt.text(0.02, 0.98, 
            f'Grid Suitable: {suitable_count}/{total_count} models\n'
            f'Best MCC: {evaluation_df["test_mcc"].max():.3f}\n'
            f'Fastest: {evaluation_df["single_latency_ms"].min():.2f}ms',
            transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_df, figsize=(15, 10)):
    """
    Plot comprehensive model comparison
    
    Args:
        results_df: DataFrame with evaluation results
        figsize: Figure size
    """
    if results_df.empty:
        print("No results to plot")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_mcc', 'test_auc']
    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC', 'AUC']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if metric in results_df.columns:
            # Sort by metric for better visualization
            sorted_data = results_df.sort_values(metric, ascending=True)
            
            bars = axes[i].barh(range(len(sorted_data)), sorted_data[metric])
            axes[i].set_yticks(range(len(sorted_data)))
            axes[i].set_yticklabels(sorted_data['model'])
            axes[i].set_xlabel(label)
            axes[i].set_title(f'Model Comparison - {label}')
            axes[i].grid(True, alpha=0.3)
            
            # Color bars based on performance
            max_val = sorted_data[metric].max()
            for j, bar in enumerate(bars):
                if sorted_data[metric].iloc[j] == max_val:
                    bar.set_color('gold')
                else:
                    bar.set_color('skyblue')
            
            # Add value labels on bars
            for j, (idx, row) in enumerate(sorted_data.iterrows()):
                axes[i].text(row[metric] + 0.01, j, f'{row[metric]:.3f}', 
                           va='center', fontsize=9)
    
    plt.tight_layout()
    plt.suptitle('Model Performance Comparison', y=1.02, fontsize=16)
    plt.show()

def plot_confusion_matrices(data_splits, results,models_to_plot=None, figsize=(15, 10)):
    """
    Plot confusion matrices for selected models
    
    Args:
        data_splits: Dictionary containing data splits
        models_to_plot: List of model names to plot (None for all)
        figsize: Figure size
    """
    if models_to_plot is None:
        models_to_plot = list(results.keys())
    
    # Filter out models with errors and limit to available models
    valid_models = [name for name in models_to_plot if name in results and 'error' not in results[name]]
    
    if not valid_models:
        print("No valid models to plot")
        return
        
    n_models = len(valid_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_models > 1 else [axes]
    else:
        axes = axes.flatten()
    
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    
    for i, model_name in enumerate(valid_models):
        if i >= len(axes):
            break
            
        result = results[model_name]
        
        try:
            if 'y_pred_val' in result:
                y_pred = result['y_pred_val']
                y_true = y_val if len(result['y_pred_val']) == len(y_val) else data_splits['y_val']
            else:
                model = result['model']
                y_pred = model.predict(X_val)
                y_true = y_val
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Fault'], 
                       yticklabels=['Normal', 'Fault'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name}\nMCC: {result.get("val_mcc", 0):.3f}\nPR-AUC: {result.get("val_pr_auc", 0):.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{model_name} - Error')
    
    # Hide empty subplots
    for i in range(len(valid_models), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Confusion Matrices', y=1.02, fontsize=16)
    plt.show()

def plot_roc_curves(data_splits, results,models_to_plot=None, figsize=(12, 10)):
    """
    Plot ROC curves for selected models
    
    Args:
        data_splits: Dictionary containing data splits
        models_to_plot: List of model names to plot (None for all)
        figsize: Figure size
    """
    if models_to_plot is None:
        models_to_plot = list(results.keys())
    
    plt.figure(figsize=figsize)
    
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    
    for model_name in models_to_plot:
        if model_name not in results or 'error' in results[model_name]:
            continue
            
        result = results[model_name]
        
        try:
            if 'y_pred_proba_val' in result:
                y_pred_proba = result['y_pred_proba_val']
                y_true = y_val if len(result['y_pred_proba_val']) == len(y_val) else data_splits['y_val']
            else:
                model = result['model']
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
                y_true = y_val
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
            
        except Exception as e:
            print(f"Error plotting ROC for {model_name}: {e}")
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def get_feature_importance(model_name, results,feature_names, top_n=20):
    """
    Get feature importance for tree-based models
    
    Args:
        model_name: Name of the model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance DataFrame
    """
    if model_name not in results or 'error' in results[model_name]:
        print(f"Model {model_name} not available")
        return pd.DataFrame()
    
    model = results[model_name]['model']
    
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        print(f"Feature importance not available for {model_name}")
        return pd.DataFrame()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df

def plot_feature_importance(model_name, results,feature_names, top_n=20, figsize=(12, 8)):
    """
    Plot feature importance for a specific model
    
    Args:
        model_name: Name of the model
        feature_names: List of feature names
        top_n: Number of top features to plot
        figsize: Figure size
    """
    importance_df = get_feature_importance(model_name, results,feature_names, top_n)
    
    if importance_df.empty:
        return
    
    plt.figure(figsize=figsize)
    
    # Create horizontal bar plot
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return importance_df
    
def plot_multiple_pr_curves(y_true_list, y_scores_list, model_names, 
                          title="PR-AUC Curves Comparison",
                          figsize=(12, 8), save_path=None, pos_label=1,
                          show_baseline=True):
    """
    Plot multiple Precision-Recall curves with AUC scores in a single graph.
    
    Parameters:
    -----------
    y_true_list : list of array-like
        List of true binary labels for each model (can be the same y_true for all models)
    y_scores_list : list of array-like  
        List of predicted probabilities/scores for each model
    model_names : list of str
        List of model names for legend
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot
    pos_label : int, default=1
        Label of the positive class
    show_baseline : bool, default=True
        Whether to show baseline (random classifier)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes  
        The axes object
    auc_scores : dict
        Dictionary of model names and their AUC scores
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    auc_scores = {}
    
    # Plot each model's PR curve
    for i, (y_true, y_scores, name) in enumerate(zip(y_true_list, y_scores_list, model_names)):
        # Calculate precision, recall, and AUC
        precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
        auc_score = average_precision_score(y_true, y_scores, pos_label=pos_label)
        auc_scores[name] = auc_score
        
        # Plot the curve
        color = colors[i % len(colors)]
        ax.plot(recall, precision, color=color, lw=2.5, 
                label=f'{name} (AUC = {auc_score:.3f})', alpha=0.8)
    
    # Show baseline (random classifier performance)
    if show_baseline and len(y_true_list) > 0:
        baseline = np.sum(y_true_list[0] == pos_label) / len(y_true_list[0])
        ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Random Baseline (AP = {baseline:.3f})')
    
    # Formatting
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    # Legend
    ax.legend(loc='lower left', fontsize=11, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.9)
    
    # Add ranking text box
    sorted_models = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
    ranking_text = "Model Ranking:\n" + "\n".join([f"{i+1}. {name}: {score:.3f}" 
                                                   for i, (name, score) in enumerate(sorted_models)])
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.98, 0.02, ranking_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    return fig, ax, auc_scores