"""
This module contains the helper function which is used for model explanations using shapley

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, Union, Any, Dict
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    shap.initjs()
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Additional interpretation imports
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_shap_explainer(model, X_background, model_tier=None, max_background_size=100):
    """
    Create SHAP explainer for a trained model
    
    Args:
        model: Trained model
        X_background: Background dataset for SHAP explainer
        model_tier: Model performance tier (optional)
        max_background_size: Maximum background size for performance
        
    Returns:
        SHAP explainer object
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available")
        return None
        
    print(f"Creating SHAP explainer...")
    
    # Limit background size for performance
    if len(X_background) > max_background_size:
        X_background_sample = X_background.sample(n=max_background_size, random_state=42)
        print(f"Using background sample of {max_background_size} points")
    else:
        X_background_sample = X_background
        
    try:
        # Choose explainer based on model type and tier
        if model_tier and 'Tier 1' in str(model_tier):
            # For ultra-fast models, use faster explainers
            if hasattr(model, 'coef_'):
                explainer = shap.LinearExplainer(model, X_background_sample)
                print("Using LinearExplainer for ultra-fast model")
            else:
                background_tiny = X_background_sample.sample(n=min(50, len(X_background_sample)), random_state=42)
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x),
                    background_tiny
                )
                print("Using KernelExplainer with small background")
        else:
            # For other models, use appropriate explainer
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_') or \
               'xgb' in str(type(model)).lower() or 'lgb' in str(type(model)).lower() or \
               'catboost' in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
                print("Using TreeExplainer for tree-based model")
            elif hasattr(model, 'coef_'):
                explainer = shap.LinearExplainer(model, X_background_sample)
                print("Using LinearExplainer")
            else:
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x),
                    X_background_sample
                )
                print("Using KernelExplainer")
                
        return explainer
        
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}")
        # Fallback to kernel explainer
        try:
            background_fallback = X_background_sample.sample(n=min(30, len(X_background_sample)), random_state=42)
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x),
                background_fallback
            )
            print("Using fallback KernelExplainer")
            return explainer
        except Exception as e2:
            print(f"Failed to create fallback explainer: {e2}")
            return None

def calculate_shap_values(explainer, X_explain, model_tier=None, max_explain_size=200):
    """
    Calculate SHAP values with optimization for different model tiers
    
    Args:
        explainer: SHAP explainer object
        X_explain: Dataset to explain
        model_tier: Model performance tier (optional)
        max_explain_size: Maximum number of samples to explain
        
    Returns:
        tuple: (shap_values, X_explain_sample)
    """
    if not SHAP_AVAILABLE or explainer is None:
        print("SHAP explainer not available")
        return None, None
        
    print("Calculating SHAP values...")
    
    # Limit explanation size based on model tier
    if model_tier and 'Tier 1' in str(model_tier):
        explain_size = min(50, len(X_explain), max_explain_size)
    elif model_tier and 'Tier 2' in str(model_tier):
        explain_size = min(100, len(X_explain), max_explain_size)
    else:
        explain_size = min(max_explain_size, len(X_explain))
    
    X_explain_sample = X_explain.sample(n=explain_size, random_state=42)
    print(f"Explaining {explain_size} samples")
    
    try:
        if isinstance(explainer, shap.KernelExplainer):
            # Optimize kernel explainer based on model tier
            if model_tier and 'Tier 1' in str(model_tier):
                n_samples = 50
            elif model_tier and 'Tier 2' in str(model_tier):
                n_samples = 100
            else:
                n_samples = 200
                
            shap_values = explainer.shap_values(X_explain_sample, nsamples=n_samples)
        else:
            shap_values = explainer.shap_values(X_explain_sample)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        print(f"SHAP values calculated. Shape: {shap_values.shape}")
        return shap_values, X_explain_sample
        
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None, None

def plot_shap_summary(shap_values, X_explain, feature_names, model_name="Model", 
                     plot_type='dot', max_display=15, figsize=(12, 8)):
    """
    Create SHAP summary plot
    
    Args:
        shap_values: SHAP values array
        X_explain: Explanation dataset
        feature_names: List of feature names
        model_name: Name of the model
        plot_type: Type of plot ('dot', 'bar', 'violin')
        max_display: Maximum number of features to display
        figsize: Figure size
    """
    if not SHAP_AVAILABLE or shap_values is None:
        print("SHAP values not available")
        return
        
    plt.figure(figsize=figsize)
    
    try:
        shap.summary_plot(shap_values, X_explain, 
                         feature_names=feature_names,
                         plot_type=plot_type, max_display=max_display, show=False)
        
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating SHAP summary plot: {e}")

def analyze_feature_importance_by_tier(shap_values, feature_names, feature_tier_mapping=None):
    """
    Analyze feature importance grouped by computational tier
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        feature_tier_mapping: Dictionary mapping features to tiers
        
    Returns:
        pd.DataFrame: Feature importance by tier
    """
    if shap_values is None:
        print("SHAP values not available")
        return pd.DataFrame()
    
    if feature_tier_mapping is None:
        feature_tier_mapping = create_feature_tier_mapping(feature_names)
    
    try:
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame with tier information
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance,
            'tier': [feature_tier_mapping.get(f, 'Unknown') for f in feature_names]
        }).sort_values('importance', ascending=False)
        
        # Group by tier
        tier_summary = importance_df.groupby('tier').agg({
            'importance': ['mean', 'sum', 'count']
        }).round(4)
        tier_summary.columns = ['mean_importance', 'total_importance', 'feature_count']
        
        print("Feature Importance by Computational Tier:")
        print("="*50)
        print(tier_summary)
        
        print(f"\nTier Efficiency Analysis:")
        for tier in tier_summary.index:
            features_in_tier = importance_df[importance_df['tier'] == tier]
            if len(features_in_tier) > 0:
                efficiency = tier_summary.loc[tier, 'total_importance'] / tier_summary.loc[tier, 'feature_count']
                print(f"{tier}: {efficiency:.4f} importance per feature")
        
        return importance_df
        
    except Exception as e:
        print(f"Error analyzing feature importance by tier: {e}")
        return pd.DataFrame()

def plot_tier_based_feature_importance(importance_df, model_name="Model", figsize=(15, 10)):
    """
    Plot feature importance grouped by computational tier
    
    Args:
        importance_df: DataFrame from analyze_feature_importance_by_tier
        model_name: Name of the model
        figsize: Figure size
    """
    if importance_df.empty:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Define tier colors
    tier_colors = {
        'Tier 1: Ultra-fast (<1ms)': 'green',
        'Tier 2: Fast (1-5ms)': 'blue',
        'Tier 3: Advanced (5-20ms)': 'orange', 
        'Power System Specific': 'red',
        'Unknown': 'gray'
    }
    
    # Plot 1: Overall feature importance
    top_features = importance_df.head(15)
    colors = [tier_colors.get(tier, 'gray') for tier in top_features['tier']]
    
    axes[0].barh(range(len(top_features)), top_features['importance'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'], fontsize=9)
    axes[0].set_xlabel('Mean Absolute SHAP Value')
    axes[0].set_title('Top 15 Most Important Features')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Importance by tier (pie chart)
    tier_totals = importance_df.groupby('tier')['importance'].sum()
    tier_colors_list = [tier_colors.get(tier, 'gray') for tier in tier_totals.index]
    
    axes[1].pie(tier_totals.values, labels=tier_totals.index, autopct='%1.1f%%',
               colors=tier_colors_list, startangle=90)
    axes[1].set_title('Total Importance by Tier')
    
    # Plot 3: Feature count by tier
    tier_counts = importance_df.groupby('tier').size()
    axes[2].bar(range(len(tier_counts)), tier_counts.values, 
               color=[tier_colors.get(tier, 'gray') for tier in tier_counts.index], alpha=0.7)
    axes[2].set_xticks(range(len(tier_counts)))
    axes[2].set_xticklabels(tier_counts.index, rotation=45, ha='right')
    axes[2].set_ylabel('Number of Features')
    axes[2].set_title('Feature Count by Tier')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Efficiency by tier
    tier_summary = importance_df.groupby('tier').agg({
        'importance': ['sum', 'count']
    })
    tier_summary.columns = ['total_importance', 'feature_count']
    tier_summary['efficiency'] = tier_summary['total_importance'] / tier_summary['feature_count']
    
    axes[3].bar(range(len(tier_summary)), tier_summary['efficiency'], 
               color=[tier_colors.get(tier, 'gray') for tier in tier_summary.index], alpha=0.7)
    axes[3].set_xticks(range(len(tier_summary)))
    axes[3].set_xticklabels(tier_summary.index, rotation=45, ha='right')
    axes[3].set_ylabel('Importance per Feature')
    axes[3].set_title('Tier Efficiency (Importance/Feature)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Feature Analysis by Computational Tier - {model_name}', y=1.02, fontsize=16)
    plt.show()

def explain_single_prediction(model, shap_explainer, shap_values, X_explain, 
                             feature_names, instance_idx=0, model_name="Model"):
    """
    Generate explanation for a single prediction
    
    Args:
        model: Trained model
        shap_explainer: SHAP explainer object
        shap_values: SHAP values array
        X_explain: Explanation dataset
        feature_names: List of feature names
        instance_idx: Index of instance to explain
        model_name: Name of the model
        
    Returns:
        dict: Explanation details
    """
    if shap_values is None:
        print("SHAP values not available")
        return {}
    
    instance_data = X_explain.iloc[instance_idx]
    instance_shap = shap_values[instance_idx]
    
    # Model prediction
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(instance_data.values.reshape(1, -1))[0]
        prediction = model.predict(instance_data.values.reshape(1, -1))[0]
        confidence = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
    else:
        prediction = model.predict(instance_data.values.reshape(1, -1))[0]
        confidence = abs(prediction)
    
    # Feature contributions
    feature_contributions = pd.DataFrame({
        'feature': feature_names,
        'value': instance_data.values,
        'shap_value': instance_shap,
        'contribution': ['FAULT' if shap_val > 0 else 'NORMAL' for shap_val in instance_shap]
    }).sort_values('shap_value', key=abs, ascending=False)
    
    # Create explanation
    explanation = {
        'instance_idx': instance_idx,
        'model_name': model_name,
        'prediction': int(prediction),
        'prediction_label': 'FAULT' if prediction == 1 else 'NORMAL',
        'confidence': float(confidence),
        'top_features': feature_contributions.head(10).to_dict('records')
    }
    
    # Print explanation
    print("=" * 60)
    print(f"PREDICTION EXPLANATION - INSTANCE {instance_idx}")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Prediction: {explanation['prediction_label']} (Confidence: {confidence:.3f})")
    print()
    
    print("TOP 10 CONTRIBUTING FEATURES:")
    print("-" * 40)
    for feat in feature_contributions.head(10).itertuples():
        direction = "→ FAULT" if feat.shap_value > 0 else "→ NORMAL"
        print(f"{feat.feature:<20}: {feat.shap_value:+.4f} {direction}")
        print(f"{'':>20}  Value: {feat.value:.4f}")
    
    return explanation

def calculate_permutation_importance_by_tier(model, X, y, feature_names, 
                                           feature_tier_mapping=None, n_repeats=5):
    """
    Calculate permutation importance grouped by computational tier
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        feature_tier_mapping: Dictionary mapping features to tiers
        n_repeats: Number of permutation repeats
        
    Returns:
        pd.DataFrame: Permutation importance by tier
    """
    if feature_tier_mapping is None:
        feature_tier_mapping = create_feature_tier_mapping(feature_names)
    
    try:
        print("Calculating permutation importance by tier...")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats,
            random_state=42, scoring='matthews_corrcoef'
        )
        
        # Create DataFrame with tier information
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'tier': [feature_tier_mapping.get(f, 'Unknown') for f in feature_names]
        }).sort_values('importance_mean', ascending=False)
        
        # Analyze by tier
        tier_analysis = importance_df.groupby('tier').agg({
            'importance_mean': ['mean', 'sum', 'count'],
            'importance_std': 'mean'
        }).round(4)
        
        print("Permutation Importance by Tier:")
        print("="*40)
        print(tier_analysis)
        
        return importance_df
        
    except Exception as e:
        print(f"Error calculating permutation importance by tier: {e}")
        return pd.DataFrame()

    
def visualize_single_shap_explanation(
    explainer: shap.Explainer,
    instance: Union[np.ndarray, pd.Series, pd.DataFrame],
    model: Any = None,
    shap_values = None,
    X_explain = None,
    instance_id = 0,
    feature_names: Optional[list] = None,
    class_index: Optional[int] = None,
    plot_type: str = 'waterfall',
    max_display: int = 10,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    show_data: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize SHAP explanation for a single instance using various plot types.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        Pre-fitted SHAP explainer object
    instance : array-like
        Single instance to explain (1D array, pandas Series, or single-row DataFrame)
    model : Any, optional
        Model object (needed for some explainer types)
    feature_names : list, optional
        Names of features. If None, will try to extract from pandas objects
    class_index : int, optional
        For multi-class classification, specify which class to explain
    plot_type : str, default 'waterfall'
        Type of plot: 'waterfall', 'force', 'bar', 'decision', or 'all'
    max_display : int, default 10
        Maximum number of features to display
    title : str, optional
        Custom title for the plot
    figsize : tuple, default (10, 6)
        Figure size for matplotlib plots
    show_data : bool, default True
        Whether to show feature values in the plot
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None (displays plots)
    
   
    """
    
    # Prepare the instance
    if isinstance(instance, pd.DataFrame):
        if len(instance) != 1:
            raise ValueError("DataFrame must contain exactly one row")
        feature_names = feature_names or instance.columns.tolist()
        instance_array = instance.values.flatten()
    elif isinstance(instance, pd.Series):
        feature_names = feature_names 
        instance_array = instance.values
    else:
        instance_array = np.array(instance).flatten()
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(instance_array))]
    
    # Reshape for explainer if needed
    if len(instance_array.shape) == 1:
        instance_reshaped = instance_array.reshape(1, -1)
    else:
        instance_reshaped = instance_array
    
    try:
        # Get SHAP values
        if hasattr(explainer, 'shap_values'):
            # For older SHAP versions or specific explainer types
            shap_values = explainer.shap_values(instance_reshaped)
            if isinstance(shap_values, list):
                # Multi-class case
                if class_index is not None:
                    shap_values = shap_values[class_index]
                else:
                    shap_values = shap_values[0]  # Default to first class
                    print(f"Multi-class detected. Showing class 0. Use class_index parameter for other classes.")
        else:
            # For newer SHAP versions
            shap_values = explainer(instance_reshaped)
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) > 2:
                    # Multi-class case
                    if class_index is not None:
                        shap_values = shap_values[:, :, class_index]
                    else:
                        shap_values = shap_values[:, :, 0]
                        print(f"Multi-class detected. Showing class 0. Use class_index parameter for other classes.")
    
    except Exception as e:
        print(f"Error getting SHAP values: {e}")
        return
    


    # explanation = explain_single_prediction(model, explainer, shap_values, X_explain, feature_names, instance_id)

    # Convert to SHAP Explanation object for newer API compatibility
    try:
        if not isinstance(shap_values, shap.Explanation):
            if hasattr(shap, 'Explanation'):
                explanation = shap.Explanation(
                    values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                    base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                    data=instance_array,
                    feature_names=feature_names
                )
            else:
                explanation = shap_values
        else:
            explanation = shap_values
    except:
        explanation = shap_values


    # Create visualizations based on plot_type
    if plot_type == 'all':
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(title or 'SHAP Explanations for Single Instance', fontsize=16)
        
        # Waterfall plot
        plt.subplot(2, 2, 1)
        try:
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            plt.title('Waterfall Plot')
        except:
            plt.title('Waterfall Plot (Not Available)')
            plt.text(0.5, 0.5, 'Waterfall plot not supported\nfor this explainer type', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Force plot (as matplotlib)
        plt.subplot(2, 2, 2)
        try:
            shap.plots.force(explanation, matplotlib=True, show=False, figsize=(8, 3))
            plt.title('Force Plot')
        except:
            plt.title('Force Plot (Not Available)')
            plt.text(0.5, 0.5, 'Force plot not supported\nfor this explainer type', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Bar plot
        plt.subplot(2, 2, 3)
        try:
            if hasattr(explanation, 'values'):
                values = explanation.values
            else:
                values = explanation[0] if len(explanation.shape) > 1 else explanation
            
            # Get top features
            abs_values = np.abs(values)
            top_indices = np.argsort(abs_values)[-max_display:]
            
            colors = ['red' if v < 0 else 'blue' for v in values[top_indices]]
            plt.barh(range(len(top_indices)), values[top_indices], color=colors)
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
            plt.xlabel('SHAP Value')
            plt.title('Bar Plot')
        except Exception as e:
            plt.title('Bar Plot (Error)')
            plt.text(0.5, 0.5, f'Error creating bar plot:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Feature values
        plt.subplot(2, 2, 4)
        if show_data:
            try:
                top_features = [feature_names[i] for i in np.argsort(np.abs(values))[-max_display:]]
                top_values = [instance_array[feature_names.index(f)] for f in top_features]
                
                plt.barh(range(len(top_features)), top_values, alpha=0.7, color='gray')
                plt.yticks(range(len(top_features)), top_features)
                plt.xlabel('Feature Value')
                plt.title('Feature Values')
            except:
                plt.title('Feature Values (Not Available)')
        
        plt.tight_layout()
        
    else:
        plt.figure(figsize=figsize)
        
        if plot_type == 'waterfall':
            try:
                shap.plots.waterfall(explanation, max_display=max_display, show=False)
                if title:
                    plt.title(title)
            except Exception as e:
                plt.text(0.5, 0.5, f'Waterfall plot error:\n{str(e)}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                
        elif plot_type == 'force':
            try:
                shap.plots.force(explanation, matplotlib=True, show=False)
                if title:
                    plt.title(title)
            except Exception as e:
                print(f"Force plot error: {e}")
                print("Trying alternative force plot...")
                try:
                    # Alternative approach for older SHAP versions
                    shap.force_plot(
                        explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                        shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                        instance_array,
                        feature_names=feature_names,
                        matplotlib=True
                    )
                except:
                    plt.text(0.5, 0.5, 'Force plot not available', ha='center', va='center')
                    
        elif plot_type == 'bar':
            try:
                if hasattr(explanation, 'values'):
                    values = explanation.values
                else:
                    values = explanation[0] if len(explanation.shape) > 1 else explanation
                
                abs_values = np.abs(values)
                top_indices = np.argsort(abs_values)[-max_display:]
                
                colors = ['red' if v < 0 else 'blue' for v in values[top_indices]]
                plt.barh(range(len(top_indices)), values[top_indices], color=colors)
                plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
                plt.xlabel('SHAP Value')
                plt.title(title or 'SHAP Values - Bar Plot')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
            except Exception as e:
                plt.text(0.5, 0.5, f'Bar plot error:\n{str(e)}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                
        elif plot_type == 'decision':
            try:
                shap.decision_plot(
                    explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                    shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                    instance_array,
                    feature_names=feature_names
                )
                if title:
                    plt.title(title)
            except Exception as e:
                plt.text(0.5, 0.5, f'Decision plot error:\n{str(e)}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

