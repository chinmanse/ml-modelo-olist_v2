import pandas as pd
import numpy as np

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable 'delivery_time_days'.
    Formula: order_estimated_delivery_date - order_delivered_customer_date
    """
    # Ensure columns are datetime
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    
    # Calculate difference in days
    df['delivery_time_days'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days
    
    return df

def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Extract features from a date column.
    """
    if date_col in df.columns:
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_day'] = df[date_col].dt.day
        df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering function.
    """
    df = create_target(df)
    
    # Extract features from purchase timestamp
    df = extract_date_features(df, 'order_purchase_timestamp')
    
    # Drop rows where target is NaN (since we can't train on them)
    df = df.dropna(subset=['delivery_time_days'])
    
    return df
