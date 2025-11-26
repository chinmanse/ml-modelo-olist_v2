import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return metrics.
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def evaluate_and_save_best_model(models_dict, X_test, y_test, output_dir='models'):
    """
    Evaluate multiple models, print metrics, and save the best one.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    best_model_name = None
    best_model_score = float('inf') # Minimizing RMSE
    best_model = None
    
    results = {}
    
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"{name} Metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        
        if metrics['rmse'] < best_model_score:
            best_model_score = metrics['rmse']
            best_model_name = name
            best_model = model
            
    print(f"Best model is {best_model_name} with RMSE: {best_model_score:.4f}")
    
    # Save best model
    model_path = os.path.join(output_dir, f'best_model_{best_model_name}.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    
    return results
