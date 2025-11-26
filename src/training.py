import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def train_xgboost(X_train, y_train):
    """
    Train XGBoost Regressor with hyperparameter tuning.
    """
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=10, 
        scoring='neg_mean_squared_error', 
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best XGBoost Params: {random_search.best_params_}")
    return random_search.best_estimator_

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Regressor.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_models(df: pd.DataFrame, target_col: str = 'delivery_time_days', feature_list: list = None):
    """
    Train models and return them.
    """
    if feature_list:
        X = df[feature_list]
    else:
        X = df.drop(columns=[target_col]).select_dtypes(include=['number'])
        
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    return {'xgboost': xgb_model, 'random_forest': rf_model}, X_test, y_test
