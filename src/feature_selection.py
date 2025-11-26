import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import os

def calculate_mutual_information(X, y):
    """
    Calculate Mutual Information for feature selection.
    """
    # Sample if data is too large
    if len(X) > 10000:
        X_sample = X.sample(10000, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y
        
    # Handle non-numeric columns by dropping them for MI calculation or encoding
    # Assuming preprocessing already encoded categorical variables or we drop them here
    X_numeric = X_sample.select_dtypes(include=[np.number])
    
    mi = mutual_info_regression(X_numeric.fillna(0), y_sample, random_state=42)
    mi_series = pd.Series(mi, index=X_numeric.columns).sort_values(ascending=False)
    return mi_series

def calculate_feature_importance_rf(X, y):
    """
    Calculate feature importance using Random Forest.
    """
    # Sample if data is too large
    if len(X) > 10000:
        X_sample = X.sample(10000, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y
        
    X_numeric = X_sample.select_dtypes(include=[np.number])
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_numeric.fillna(0), y_sample)
    
    importances = pd.Series(rf.feature_importances_, index=X_numeric.columns).sort_values(ascending=False)
    return importances

def select_features(df: pd.DataFrame, target_col: str = 'delivery_time_days', top_n: int = 20) -> list:
    """
    Select top features based on MI and RF importance.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Drop non-numeric columns for feature selection (or assume they are encoded)
    # For now, we'll stick to numeric features for simplicity and robustness
    X = X.select_dtypes(include=[np.number])
    
    print("Calculating Mutual Information...")
    mi_scores = calculate_mutual_information(X, y)
    
    print("Calculating Random Forest Importance...")
    rf_scores = calculate_feature_importance_rf(X, y)
    
    # Combine scores (simple average of normalized scores)
    # Normalize
    mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    rf_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min())
    
    combined_score = (mi_norm + rf_norm) / 2
    combined_score = combined_score.sort_values(ascending=False)
    
    top_features = combined_score.head(top_n).index.tolist()
    
    print(f"Top {top_n} features selected: {top_features}")
    
    # Save to CSV
    pd.DataFrame({'feature': top_features}).to_csv('data/processed/feature_shortlist.csv', index=False)
    
    return top_features
