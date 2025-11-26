import pandas as pd
import numpy as np

def convert_dates(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """
    Convert specified columns to datetime objects.
    """
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values.
    For simplicity and based on the notebook, we might drop rows with missing target
    or important dates, and fill others.
    """
    # Based on notebook logic:
    # mask = y.notna() -> handled in feature engineering usually when defining target
    # But here we can do general cleaning.
    
    # For numerical columns, fill with median (as per notebook)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # For categorical, fill with 'Unknown' or mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        
    return df

def encode_categorical(df: pd.DataFrame, max_cardinality: int = 100) -> pd.DataFrame:
    """
    One-hot encode categorical columns with low cardinality.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_small = [c for c in cat_cols if df[c].nunique() <= max_cardinality]
    
    df = pd.get_dummies(df, columns=cat_small, dummy_na=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing function.
    """
    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'shipping_limit_date'
    ]
    
    df = convert_dates(df, date_cols)
    # Note: Missing values handling might be better done after feature engineering 
    # if we need to calculate differences between dates that might be null.
    # However, the notebook fills NA before model training.
    # We will defer full NA handling to after feature engineering for target creation,
    # but we can do some basic cleaning here if needed.
    
    return df
