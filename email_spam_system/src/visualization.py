import matplotlib
# CRITICAL FIX: Force non-interactive backend to prevent Tcl/Tk errors on Windows
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc

def plot_feature_distributions(df: pd.DataFrame, target_col: str):
    """
    Plots histograms overlaying Spam vs Ham distributions for the manual features.
    Saves to 'data/feature_distributions.png' instead of opening a window.
    """
    features = ['body_len', 'punct%', 'cap_count']
    
    # Setup the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle('Feature Distributions: Spam vs Ham', fontsize=16)
    
    for i, feature in enumerate(features):
        # Check if column exists to avoid errors
        if feature not in df.columns:
            continue
            
        # Create dynamic bins for better visualization
        bins = np.linspace(0, 200, 40) if feature != 'body_len' else 40
        
        sns.histplot(data=df, x=feature, hue=target_col, kde=True, ax=axes[i], palette="bright", bins=bins)
        axes[i].set_title(f'Distribution of {feature}')
        
        # Limit x-axis for body_len to avoid squishing the plot due to outliers
        if feature == 'body_len':
            axes[i].set_xlim(0, 2000) # Limit x-axis for readability
            
    plt.tight_layout()
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # SAVE the plot to disk instead of showing it
    output_path = 'data/feature_distributions.png'
    plt.savefig(output_path)
    print(f" Plot saved successfully to: {output_path}")
    
    # Close the plot to free memory
    plt.close()

def plot_correlation(df: pd.DataFrame):
    """
    Saves a correlation heatmap of the engineered features.
    """
    if df.empty: return

    plt.figure(figsize=(8,6))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Feature Correlation Matrix")
        
        output_path = 'data/feature_correlation.png'
        plt.savefig(output_path)
        print(f"   Correlation matrix saved to: {output_path}")
        plt.close()

def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plots ROC curves for multiple models on a single graph.
    Input:
        models_dict: Dictionary of {name: trained_pipeline}
        X_test: Test features
        y_test: Test labels (encoded as 0/1)
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        # Get probabilities for the positive class (Spam)
        try:
            # Some models (like SVM) need probability=True set during initialization
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                # Fallback for models that might not have predict_proba (though our SVM/RF/NB do)
                print(f"Model {name} does not support probability prediction. Skipping ROC.")
                continue
        except AttributeError:
            print(f"Model {name} encountered error in probability prediction.")
            continue
            
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Model Comparison')
    plt.legend(loc="lower right")
    
    output_path = 'data/roc_curve_comparison.png'
    plt.savefig(output_path)
    print(f"   ROC Curve saved to: {output_path}")
    plt.close()