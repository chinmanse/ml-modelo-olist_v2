import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_model_performance(results_df):
    """
    Plot model performance metrics.
    """
    metrics = ['RMSE', 'MAE', 'R2']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        sns.barplot(x='Model', y=metric, data=results_df, ax=axes[i])
        axes[i].set_title(f'Model Comparison - {metric}')
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()

def plot_predictions(y_test, y_pred, model_name):
    """
    Plot Actual vs Predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """
    Plot feature importance.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png')
        plt.close()
    else:
        print(f"Model {model_name} does not have feature_importances_ attribute.")
