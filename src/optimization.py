import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_baseline_model(X_train, y_train):
    """
    Train a baseline Linear Regression model.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest hyperparameters.
    """
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    random_search = RandomizedSearchCV(
        rf, 
        param_distributions=param_dist, 
        n_iter=2, 
        scoring='neg_root_mean_squared_error', 
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best RF Params: {random_search.best_params_}")
    return random_search.best_estimator_

def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost hyperparameters.
    """
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    param_dist = {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [0, 0.1, 1, 10]
    }
    
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=2, 
        scoring='neg_root_mean_squared_error', 
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best XGBoost Params: {random_search.best_params_}")
    return random_search.best_estimator_

def compare_models(models, X_test, y_test):
    """
    Compare models and return a DataFrame of results.
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    
    return pd.DataFrame(results)
