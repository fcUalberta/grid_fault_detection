"""
The module contains the set of helper function that will help us create trends analysis, visualizations, use small language model to explain the results and finally integrate them in a PDF report to be saved to the archive/sent via email.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Rect
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("reportlab not available. Install with: pip install reportlab")

# For open-source language models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformers not available. Install with: pip install transformers torch")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def setup_language_model(model_type="huggingface", model_name="microsoft/DialoGPT-small"):
    """
    Setup open-source language model for report generation
    
    Parameters:
    -----------
    model_type : str
        Either "huggingface" for Hugging Face models or "ollama" for local Ollama models
    model_name : str
        Model name/identifier
        
    Returns:
    --------
    generator : object
        Language model pipeline or client
    """
    if model_type == "huggingface" and TRANSFORMERS_AVAILABLE:
        try:
            # Use a small, efficient model for report generation
            generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            print(f"Successfully loaded Hugging Face model: {model_name}")
            return generator
        except Exception as e:
            print(f"Error loading Hugging Face model: {e}")
            return None
            
    elif model_type == "ollama" and OLLAMA_AVAILABLE:
        try:
            # Test connection to Ollama
            response = ollama.list()
            available_models = [model['name'] for model in response['models']]
            
            if model_name in available_models:
                print(f"Successfully connected to Ollama model: {model_name}")
                return ollama
            else:
                print(f"Model {model_name} not found in Ollama. Available models: {available_models}")
                return None
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return None
    else:
        print("No suitable language model setup available")
        return None

def analyze_data_trends(predictions_df: pd.DataFrame, 
                       historical_data: Optional[pd.DataFrame] = None,
                       time_window_hours: int = 24,
                       data_source_type: str = "features") -> Dict:
    """
    Analyze trends in prediction data and historical patterns
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with prediction results. Can be:
        - Feature data: columns like signal_id, prediction, confidence, actual (optional)
        - Time series data: includes timestamp column
    historical_data : pd.DataFrame, optional
        Historical data for trend comparison
    time_window_hours : int
        Analysis window in hours (used for synthetic timestamps if needed)
    data_source_type : str
        Type of data: "features" (no timestamps) or "timeseries" (has timestamps)
        
    Returns:
    --------
    analysis : dict
        Comprehensive trend analysis
    """
    # Create a copy to avoid modifying original data
    data = predictions_df.copy()
    
    # Handle different data types
    if data_source_type == "features" or 'timestamp' not in data.columns:
        # Feature data without timestamps - create synthetic timestamps
        print("No timestamp column found. Creating synthetic timestamps for analysis...")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        data['timestamp'] = pd.date_range(
            start=start_time,
            end=end_time,
            periods=len(data)
        )
        # Use all data as current window
        current_window = data.copy()
    else:
        # Time series data with timestamps
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        end_time = data['timestamp'].max()
        start_time = end_time - timedelta(hours=time_window_hours)
        current_window = data[data['timestamp'] >= start_time].copy()
    
    analysis = {
        'time_window': {
            'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_hours': time_window_hours,
            'total_predictions': len(current_window)
        }
    }
    
    # Fault detection trends
    fault_predictions = current_window[current_window['prediction'] == 1]
    analysis['fault_trends'] = {
        'total_faults_detected': len(fault_predictions),
        'fault_rate_percent': (len(fault_predictions) / len(current_window) * 100) if len(current_window) > 0 else 0,
        'average_confidence': fault_predictions['confidence'].mean() if len(fault_predictions) > 0 else 0,
        'peak_fault_hours': []
    }
    
    # Hourly fault distribution
    current_window['hour'] = current_window['timestamp'].dt.hour
    hourly_faults = current_window.groupby('hour')['prediction'].sum()
    peak_hours = hourly_faults.nlargest(3).index.tolist()
    analysis['fault_trends']['peak_fault_hours'] = peak_hours
    
    # Signal ID patterns
    signal_fault_rates = current_window.groupby('signal_id').agg({
        'prediction': ['count', 'sum'],
        'confidence': 'mean'
    }).round(3)
    
    # Flatten column names
    signal_fault_rates.columns = ['total_predictions', 'fault_count', 'avg_confidence']
    signal_fault_rates['fault_rate'] = (signal_fault_rates['fault_count'] / 
                                       signal_fault_rates['total_predictions'] * 100).round(2)
    
    # Top problematic signals
    top_problematic = signal_fault_rates.nlargest(5, 'fault_rate')
    analysis['signal_patterns'] = {
        'most_problematic_signals': top_problematic.to_dict('index'),
        'total_unique_signals': len(signal_fault_rates),
        'signals_with_faults': len(signal_fault_rates[signal_fault_rates['fault_count'] > 0])
    }
    
    # Model performance (if actual values available)
    if 'actual' in current_window.columns:
        from sklearn.metrics import classification_report, matthews_corrcoef
        
        y_true = current_window['actual']
        y_pred = current_window['prediction']
        
        analysis['model_performance'] = {
            'accuracy': (y_true == y_pred).mean(),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'fault_detection_rate': (y_pred[y_true == 1] == 1).mean() if (y_true == 1).sum() > 0 else 0,
            'false_positive_rate': (y_pred[y_true == 0] == 1).mean() if (y_true == 0).sum() > 0 else 0
        }
    
    # Historical comparison (if available)
    if historical_data is not None:
        hist_fault_rate = historical_data['prediction'].mean() * 100
        current_fault_rate = analysis['fault_trends']['fault_rate_percent']
        
        analysis['historical_comparison'] = {
            'current_fault_rate': current_fault_rate,
            'historical_average': hist_fault_rate,
            'rate_change_percent': ((current_fault_rate - hist_fault_rate) / hist_fault_rate * 100) 
                                  if hist_fault_rate > 0 else 0,
            'trend': 'increasing' if current_fault_rate > hist_fault_rate else 'decreasing'
        }
    
    # Alert thresholds
    analysis['alerts'] = {
        'high_fault_rate': analysis['fault_trends']['fault_rate_percent'] > 5.0,
        'low_confidence_detections': analysis['fault_trends']['average_confidence'] < 0.7,
        'model_performance_degradation': analysis.get('model_performance', {}).get('mcc', 1.0) < 0.5
    }
    
    return analysis

def generate_llm_insights(analysis: Dict, 
                         generator, 
                         model_type: str = "huggingface") -> str:
    """
    Generate intelligent insights using language model
    
    Parameters:
    -----------
    analysis : dict
        Analysis results from analyze_data_trends
    generator : object
        Language model pipeline or client
    model_type : str
        Type of model ("huggingface" or "ollama")
        
    Returns:
    --------
    insights : str
        Generated insights and recommendations
    """
    if generator is None:
        return "Language model not available. Using template-based insights."
    
    # Create structured prompt
    prompt = f"""
    Power Grid Fault Detection Analysis Report
    
    Time Period: {analysis['time_window']['start']} to {analysis['time_window']['end']}
    Total Predictions: {analysis['time_window']['total_predictions']}
    
    Key Findings:
    - Fault Rate: {analysis['fault_trends']['fault_rate_percent']:.2f}%
    - Total Faults Detected: {analysis['fault_trends']['total_faults_detected']}
    - Average Confidence: {analysis['fault_trends']['average_confidence']:.3f}
    - Peak Fault Hours: {analysis['fault_trends']['peak_fault_hours']}
    
    Signal Analysis:
    - Signals with Faults: {analysis['signal_patterns']['signals_with_faults']} out of {analysis['signal_patterns']['total_unique_signals']}
    
    Alerts:
    - High Fault Rate: {analysis['alerts']['high_fault_rate']}
    - Low Confidence: {analysis['alerts']['low_confidence_detections']}
    
    Based on this power grid monitoring data, provide a concise professional analysis including:
    1. Key operational insights
    2. Potential concerns or anomalies
    3. Recommended actions for grid operators
    4. Maintenance priorities
    
    Analysis:"""
    
    try:
        if model_type == "huggingface":
            # Generate with Hugging Face model
            response = generator(
                prompt,
                max_length=len(prompt.split()) + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            insights = response[0]['generated_text'][len(prompt):].strip()
            
        elif model_type == "ollama":
            # Generate with Ollama
            response = generator.generate(
                model='llama2',  # or your preferred Ollama model
                prompt=prompt,
                options={'temperature': 0.7, 'max_tokens': 300}
            )
            insights = response['response'].strip()
            
        else:
            insights = "Unsupported model type"
            
        return insights
        
    except Exception as e:
        print(f"Error generating LLM insights: {e}")
        return generate_template_insights(analysis)

def generate_template_insights(analysis: Dict) -> str:
    """
    Generate template-based insights when LLM is not available
    """
    insights = []
    
    # Fault rate analysis
    fault_rate = analysis['fault_trends']['fault_rate_percent']
    if fault_rate > 5.0:
        insights.append(f"⚠️ HIGH ALERT: Fault rate of {fault_rate:.2f}% exceeds normal operational threshold (5%). Immediate investigation recommended.")
    elif fault_rate > 2.0:
        insights.append(f"⚡ ELEVATED: Fault rate of {fault_rate:.2f}% is above baseline. Monitor closely for developing issues.")
    else:
        insights.append(f"✅ NORMAL: Fault rate of {fault_rate:.2f}% is within acceptable operational range.")
    
    # Confidence analysis
    avg_confidence = analysis['fault_trends']['average_confidence']
    if avg_confidence < 0.7:
        insights.append(f"🔍 Model confidence averaging {avg_confidence:.3f} is lower than preferred. Consider model retraining or feature review.")
    
    # Peak hours
    peak_hours = analysis['fault_trends']['peak_fault_hours']
    if peak_hours:
        insights.append(f"📊 Peak fault detection hours: {peak_hours}. Schedule maintenance during off-peak periods.")
    
    # Signal patterns
    signals_with_faults = analysis['signal_patterns']['signals_with_faults']
    total_signals = analysis['signal_patterns']['total_unique_signals']
    fault_coverage = (signals_with_faults / total_signals * 100) if total_signals > 0 else 0
    
    if fault_coverage > 20:
        insights.append(f"🚨 {fault_coverage:.1f}% of monitored signals showing faults. System-wide issue possible.")
    elif fault_coverage > 10:
        insights.append(f"⚠️ {fault_coverage:.1f}% of signals affected. Regional or equipment-specific problems likely.")
    
    # Performance alerts
    if analysis['alerts']['high_fault_rate']:
        insights.append("🔴 PRIORITY: High fault rate detected - dispatch maintenance teams immediately.")
    
    if analysis['alerts']['low_confidence_detections']:
        insights.append("🟡 CAUTION: Low confidence in fault predictions - verify with manual inspection.")
    
    # Historical comparison
    if 'historical_comparison' in analysis:
        trend = analysis['historical_comparison']['trend']
        change = analysis['historical_comparison']['rate_change_percent']
        if abs(change) > 25:
            insights.append(f"📈 Fault rate {trend} by {abs(change):.1f}% compared to historical average.")
    
    return " ".join(insights)

def create_diagnostic_visualizations(analysis: Dict, 
                                   predictions_df: pd.DataFrame,
                                   output_dir: str = "diagnostics_reports",
                                   for_pdf: bool = True,
                                   data_source_type: str = "features") -> List[str]:
    """
    Create visualization plots for the diagnostic report
    
    Parameters:
    -----------
    for_pdf : bool
        If True, optimizes plots for PDF inclusion (smaller file size, better formatting)
    data_source_type : str
        Type of data: "features" (no timestamps) or "timeseries" (has timestamps)
    
    Returns:
    --------
    plot_paths : list
        Paths to saved plot files
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
    
    # Set style for PDF optimization
    if for_pdf:
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    else:
        plt.style.use('default')
    
    # Create a copy of data and handle timestamps
    data = predictions_df.copy()
    
    if data_source_type == "features" or 'timestamp' not in data.columns:
        # For feature data, create synthetic timestamps for visualization
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        data['timestamp'] = pd.date_range(
            start=start_time,
            end=end_time,
            periods=len(data)
        )
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    
    # 1. Fault Detection Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8) if for_pdf else (12, 8))
    
    # For feature data, show fault distribution by prediction order instead of time
    if data_source_type == "features":
        # Fault distribution across data samples
        fault_data = data[data['prediction'] == 1]
        normal_data = data[data['prediction'] == 0]
        
        # Create bins for fault distribution across sample indices
        total_samples = len(data)
        n_bins = min(24, total_samples // 50) if total_samples > 50 else 1
        
        if n_bins > 1:
            bin_edges = np.linspace(0, total_samples, n_bins + 1)
            fault_indices = fault_data.index.values if len(fault_data) > 0 else []
            fault_counts, _ = np.histogram(fault_indices, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            bars1 = ax1.bar(range(len(fault_counts)), fault_counts, alpha=0.7, color='#d32f2f')
            ax1.set_title('Fault Distribution Across Data Samples', fontweight='bold')
            ax1.set_xlabel('Sample Batch')
            ax1.set_ylabel('Number of Faults')
            ax1.set_xticks(range(len(fault_counts)))
            ax1.set_xticklabels([f'Batch {i+1}' for i in range(len(fault_counts))], rotation=45)
        else:
            # Single bar showing total faults vs normal
            counts = [len(normal_data), len(fault_data)]
            labels = ['Normal', 'Fault']
            colors = ['#4caf50', '#d32f2f']
            bars1 = ax1.bar(labels, counts, color=colors, alpha=0.7)
            ax1.set_title('Overall Fault vs Normal Distribution', fontweight='bold')
            ax1.set_ylabel('Number of Samples')
    else:
        # Original time-based analysis for time series data
        hourly_faults = data.groupby('hour')['prediction'].sum()
        bars1 = ax1.bar(hourly_faults.index, hourly_faults.values, alpha=0.7, color='#d32f2f')
        ax1.set_title('Fault Detections by Hour of Day', fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Faults')
    
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Confidence distribution (same for both data types)
    fault_data = data[data['prediction'] == 1]
    if len(fault_data) > 0 and 'confidence' in data.columns:
        n, bins, patches = ax2.hist(fault_data['confidence'], bins=15, alpha=0.7, 
                                   color='#ff9800', edgecolor='black', linewidth=0.5)
        ax2.axvline(fault_data['confidence'].mean(), color='#d32f2f', linestyle='--', 
                   linewidth=2, label=f'Mean: {fault_data["confidence"].mean():.3f}')
        ax2.set_title('Confidence Distribution for Fault Predictions', fontweight='bold')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    elif 'confidence' not in data.columns:
        # Show prediction distribution if no confidence data
        pred_counts = data['prediction'].value_counts().sort_index()
        bars2 = ax2.bar(['Normal (0)', 'Fault (1)'], pred_counts.values, 
                       color=['#4caf50', '#d32f2f'], alpha=0.7)
        ax2.set_title('Prediction Distribution', fontweight='bold')
        ax2.set_ylabel('Count')
        
        for bar, count in zip(bars2, pred_counts.values):
            ax2.annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    timeline_path = os.path.join(output_dir, f'fault_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(timeline_path, dpi=150 if for_pdf else 300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    plot_paths.append(timeline_path)
    
    # 2. System Overview Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8) if for_pdf else (12, 10))
    
    # Fault rate pie chart
    fault_count = analysis['fault_trends']['total_faults_detected']
    normal_count = analysis['time_window']['total_predictions'] - fault_count
    
    if fault_count > 0:
        labels = ['Normal', 'Fault']
        sizes = [normal_count, fault_count]
        colors = ['#4caf50', '#f44336']
        explode = (0, 0.1)  # explode fault slice
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.set_title('System Status Distribution', fontweight='bold')
    else:
        ax1.pie([1], labels=['Normal Operation'], colors=['#4caf50'], autopct='100%')
        ax1.set_title('System Status: Normal', fontweight='bold')
    
    # Signal analysis bar chart
    if 'signal_id' in predictions_df.columns:
        signal_analysis = predictions_df.groupby('signal_id').agg({
            'prediction': ['count', 'sum'],
            'confidence': 'mean'
        })
        signal_analysis.columns = ['total_predictions', 'fault_count', 'avg_confidence']
        signal_analysis['fault_rate'] = (signal_analysis['fault_count'] / 
                                       signal_analysis['total_predictions'] * 100)
        
        # Top 10 problematic signals
        top_signals = signal_analysis.nlargest(10, 'fault_rate')
        
        if len(top_signals) > 0:
            bars2 = ax2.bar(range(len(top_signals)), top_signals['fault_rate'], 
                           color='#ff5722', alpha=0.7)
            ax2.set_title('Top 10 Problematic Signals (Fault Rate %)', fontweight='bold')
            ax2.set_xlabel('Signal Rank')
            ax2.set_ylabel('Fault Rate (%)')
            ax2.set_xticks(range(len(top_signals)))
            ax2.set_xticklabels([f'#{i+1}' for i in range(len(top_signals))])
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    # Model performance metrics (if available)
    if 'model_performance' in analysis:
        metrics = ['Accuracy', 'MCC', 'F1', 'Precision', 'Recall']
        values = [
            analysis['model_performance'].get('accuracy', 0),
            analysis['model_performance'].get('mcc', 0),
            analysis['model_performance'].get('f1_score', 0),
            analysis['model_performance'].get('precision', 0),
            analysis['model_performance'].get('recall', 0)
        ]
        
        bars3 = ax3.bar(metrics, values, color=['#2196f3', '#9c27b0', '#ff9800', '#4caf50', '#f44336'])
        ax3.set_title('Model Performance Metrics', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, values):
            height = bar.get_height()
            ax3.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Alert status
    alert_labels = ['High Fault Rate', 'Low Confidence', 'Performance Issues']
    alert_status = [
        analysis['alerts']['high_fault_rate'],
        analysis['alerts']['low_confidence_detections'],
        analysis['alerts']['model_performance_degradation']
    ]
    alert_colors = ['#f44336' if status else '#4caf50' for status in alert_status]
    alert_values = [1 if status else 0 for status in alert_status]
    
    bars4 = ax4.bar(alert_labels, alert_values, color=alert_colors, alpha=0.7)
    ax4.set_title('System Alerts Status', fontweight='bold')
    ax4.set_ylabel('Alert Active')
    ax4.set_ylim(0, 1.2)
    ax4.set_xticklabels(alert_labels, rotation=45, ha='right')
    
    # Add status labels
    for bar, status in zip(bars4, alert_status):
        height = bar.get_height()
        label = 'ACTIVE' if status else 'NORMAL'
        ax4.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.05),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, f'system_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(dashboard_path, dpi=150 if for_pdf else 300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    plot_paths.append(dashboard_path)
    
    return plot_paths

def create_pdf_report(analysis: Dict,
                     insights: str,
                     plot_paths: List[str],
                     output_dir: str = "diagnostics_reports") -> str:
    """
    Create a professional PDF report using ReportLab
    
    Parameters:
    -----------
    analysis : dict
        Analysis results from analyze_data_trends
    insights : str
        Generated insights text
    plot_paths : list
        Paths to visualization plots
    output_dir : str
        Output directory for the PDF report
        
    Returns:
    --------
    pdf_path : str
        Path to generated PDF report
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup PDF document
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f'VSB_Diagnostics_Report_{report_timestamp}.pdf'
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=12,
        textColor=colors.darkslategray
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    alert_style = ParagraphStyle(
        'AlertStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leftIndent=20,
        rightIndent=20,
        borderColor=colors.red,
        borderWidth=1,
        borderPadding=10,
        backColor=colors.lightgrey
    )
    
    # Title and header
    story.append(Paragraph("VSB Power Line Fault Detection", title_style))
    story.append(Paragraph("Automated Diagnostics Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_data = [
        ['Report Generated:', report_time],
        ['Analysis Period:', f"{analysis['time_window']['start']} to {analysis['time_window']['end']}"],
        ['Total Predictions:', str(analysis['time_window']['total_predictions'])],
        ['Analysis Duration:', f"{analysis['time_window']['duration_hours']} hours"]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    # Alert status
    alert_text = ""
    if analysis['alerts']['high_fault_rate']:
        alert_text = "🚨 CRITICAL ALERT: High fault rate detected - Immediate attention required"
        story.append(Paragraph(alert_text, alert_style))
    elif analysis['alerts']['low_confidence_detections']:
        alert_text = "⚠️ WARNING: Low confidence in predictions - Model review recommended"
        story.append(Paragraph(alert_text, alert_style))
    elif analysis['alerts']['model_performance_degradation']:
        alert_text = "⚠️ WARNING: Model performance degradation detected"
        story.append(Paragraph(alert_text, alert_style))
    else:
        alert_text = "✅ NORMAL: All systems operating within normal parameters"
        story.append(Paragraph(alert_text, normal_style))
    
    story.append(Spacer(1, 12))
    story.append(Paragraph(insights, normal_style))
    story.append(Spacer(1, 20))
    
    # Key Metrics Section
    story.append(Paragraph("Key Performance Metrics", heading_style))
    
    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Total Faults Detected', str(analysis['fault_trends']['total_faults_detected']), 
         'HIGH' if analysis['fault_trends']['total_faults_detected'] > 10 else 'NORMAL'],
        ['Fault Rate', f"{analysis['fault_trends']['fault_rate_percent']:.2f}%",
         'HIGH' if analysis['fault_trends']['fault_rate_percent'] > 5.0 else 'NORMAL'],
        ['Average Confidence', f"{analysis['fault_trends']['average_confidence']:.3f}",
         'LOW' if analysis['fault_trends']['average_confidence'] < 0.7 else 'NORMAL'],
        ['Affected Signals', f"{analysis['signal_patterns']['signals_with_faults']} / {analysis['signal_patterns']['total_unique_signals']}",
         'HIGH' if analysis['signal_patterns']['signals_with_faults'] > analysis['signal_patterns']['total_unique_signals'] * 0.1 else 'NORMAL']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Model Performance (if available)
    if 'model_performance' in analysis:
        story.append(Paragraph("Model Performance Analysis", heading_style))
        
        perf_data = [
            ['Metric', 'Value', 'Threshold', 'Status'],
            ['Accuracy', f"{analysis['model_performance']['accuracy']:.3f}", '> 0.90', 
             'GOOD' if analysis['model_performance']['accuracy'] > 0.90 else 'REVIEW'],
            ['Matthews Correlation Coefficient', f"{analysis['model_performance']['mcc']:.3f}", '> 0.50',
             'GOOD' if analysis['model_performance']['mcc'] > 0.50 else 'REVIEW'],
            ['Fault Detection Rate', f"{analysis['model_performance']['fault_detection_rate']:.3f}", '> 0.85',
             'GOOD' if analysis['model_performance']['fault_detection_rate'] > 0.85 else 'REVIEW'],
            ['False Positive Rate', f"{analysis['model_performance']['false_positive_rate']:.3f}", '< 0.10',
             'GOOD' if analysis['model_performance']['false_positive_rate'] < 0.10 else 'REVIEW']
        ]
        
        perf_table = Table(perf_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 20))
    
    # Detailed Analysis
    story.append(Paragraph("Detailed Analysis", heading_style))
    
    story.append(Paragraph("Fault Detection Patterns", subheading_style))
    patterns_text = f"""
    • Peak fault detection hours: {', '.join(map(str, analysis['fault_trends']['peak_fault_hours']))}
    • Total signals monitored: {analysis['signal_patterns']['total_unique_signals']}
    • Signals with detected faults: {analysis['signal_patterns']['signals_with_faults']}
    • System fault coverage: {(analysis['signal_patterns']['signals_with_faults'] / analysis['signal_patterns']['total_unique_signals'] * 100):.1f}%
    """
    story.append(Paragraph(patterns_text, normal_style))
    
    # Historical comparison (if available)
    if 'historical_comparison' in analysis:
        story.append(Paragraph("Historical Trend Analysis", subheading_style))
        hist_text = f"""
        Current fault rate ({analysis['historical_comparison']['current_fault_rate']:.2f}%) compared to historical average 
        ({analysis['historical_comparison']['historical_average']:.2f}%) shows a {analysis['historical_comparison']['trend']} 
        trend with {abs(analysis['historical_comparison']['rate_change_percent']):.1f}% change.
        """
        story.append(Paragraph(hist_text, normal_style))
    
    story.append(PageBreak())
    
    # Visualizations
    story.append(Paragraph("System Visualizations", heading_style))
    
    for i, plot_path in enumerate(plot_paths):
        if os.path.exists(plot_path):
            story.append(Paragraph(f"Chart {i+1}: System Analysis", subheading_style))
            
            # Add image with proper sizing
            img = Image(plot_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
    
    # Recommendations
    story.append(PageBreak())
    story.append(Paragraph("Operational Recommendations", heading_style))
    
    recommendations = []
    
    if analysis['alerts']['high_fault_rate']:
        recommendations.append("IMMEDIATE: Dispatch maintenance teams to investigate high fault rate areas")
    
    if analysis['alerts']['low_confidence_detections']:
        recommendations.append("PRIORITY: Review model performance and consider retraining with recent data")
    
    if analysis['fault_trends']['peak_fault_hours']:
        peak_hours_str = ', '.join(map(str, analysis['fault_trends']['peak_fault_hours']))
        recommendations.append(f"SCHEDULING: Focus maintenance during peak fault hours: {peak_hours_str}")
    
    if analysis['signal_patterns']['signals_with_faults'] > analysis['signal_patterns']['total_unique_signals'] * 0.1:
        recommendations.append("INVESTIGATION: Multiple signals affected - investigate common infrastructure")
    
    if not recommendations:
        recommendations.append("MAINTENANCE: Continue routine monitoring and preventive maintenance schedule")
        recommendations.append("OPTIMIZATION: Current system performance is within acceptable parameters")
    
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", normal_style))
        story.append(Spacer(1, 6))
    
    # Technical appendix
    story.append(PageBreak())
    story.append(Paragraph("Technical Data Appendix", heading_style))
    
    # Convert analysis to formatted text
    tech_data = f"""
    Complete Analysis Data:
    
    Time Window Analysis:
    - Start Time: {analysis['time_window']['start']}
    - End Time: {analysis['time_window']['end']}
    - Duration: {analysis['time_window']['duration_hours']} hours
    - Total Predictions: {analysis['time_window']['total_predictions']}
    
    Fault Detection Results:
    - Total Faults: {analysis['fault_trends']['total_faults_detected']}
    - Fault Rate: {analysis['fault_trends']['fault_rate_percent']:.3f}%
    - Average Confidence: {analysis['fault_trends']['average_confidence']:.3f}
    
    Signal Pattern Analysis:
    - Total Unique Signals: {analysis['signal_patterns']['total_unique_signals']}
    - Signals with Faults: {analysis['signal_patterns']['signals_with_faults']}
    - Peak Fault Hours: {analysis['fault_trends']['peak_fault_hours']}
    
    Alert Status:
    - High Fault Rate Alert: {analysis['alerts']['high_fault_rate']}
    - Low Confidence Alert: {analysis['alerts']['low_confidence_detections']}
    - Performance Degradation Alert: {analysis['alerts']['model_performance_degradation']}
    """
    
    story.append(Paragraph(tech_data, normal_style))
    
    # Build PDF
    doc.build(story)
    
    print(f"PDF report generated successfully: {pdf_path}")
    return pdf_path

def generate_diagnostics_report(predictions_df: pd.DataFrame,
                              historical_data: Optional[pd.DataFrame] = None,
                              time_window_hours: int = 24,
                              output_dir: str = "diagnostics_reports",
                              use_llm: bool = True,
                              model_type: str = "huggingface",
                              model_name: str = "microsoft/DialoGPT-small",
                              output_format: str = "pdf",
                              data_source_type: str = "auto") -> str:
    """
    Generate comprehensive automated diagnostics report
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with prediction results. Expected columns:
        - For feature data: signal_id (optional), prediction, confidence (optional), actual (optional)
        - For time series: timestamp, signal_id (optional), prediction, confidence (optional), actual (optional)
    historical_data : pd.DataFrame, optional
        Historical data for comparison
    time_window_hours : int
        Analysis window in hours (used for synthetic timestamps if needed)
    output_dir : str
        Output directory for reports
    use_llm : bool
        Whether to use language model for insights
    model_type : str
        Type of language model to use
    model_name : str
        Name/identifier of the model
    output_format : str
        Output format: "pdf" or "html"
    data_source_type : str
        Type of data: "auto" (detect), "features" (no timestamps), or "timeseries" (has timestamps)
        
    Returns:
    --------
    report_path : str
        Path to generated report
    """
    # Validate input data
    required_columns = ['prediction']
    missing_columns = [col for col in required_columns if col not in predictions_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Auto-detect data source type
    if data_source_type == "auto":
        if 'timestamp' in predictions_df.columns:
            data_source_type = "timeseries"
            print("Detected time series data with timestamps")
        else:
            data_source_type = "features"
            print("Detected feature data without timestamps")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform trend analysis
    print("Analyzing data trends...")
    analysis = analyze_data_trends(predictions_df, historical_data, time_window_hours, data_source_type)
    
    # Setup language model if requested
    generator = None
    if use_llm:
        print(f"Setting up {model_type} language model...")
        generator = setup_language_model(model_type, model_name)
    
    # Generate insights
    print("Generating insights...")
    if use_llm and generator is not None:
        insights = generate_llm_insights(analysis, generator, model_type)
    else:
        insights = generate_template_insights(analysis)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_paths = create_diagnostic_visualizations(
        analysis, predictions_df, output_dir, 
        for_pdf=(output_format == "pdf"), 
        data_source_type=data_source_type
    )
    
    # Generate report based on format
    if output_format.lower() == "pdf":
        print("Generating PDF report...")
        report_path = create_pdf_report(analysis, insights, plot_paths, output_dir)
    else:
        print("Generating HTML report...")
        report_path = create_html_report(analysis, insights, plot_paths, output_dir)
    
    # Save JSON analysis
    json_filename = f'analysis_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"Diagnostics report generated successfully!")
    print(f"Report: {report_path}")
    print(f"JSON Data: {json_path}")
    print(f"Visualizations: {len(plot_paths)} plots saved")
    print(f"Data source type: {data_source_type}")
    
    return report_path


