import pandas as pd
import os
from src.ingestion import load_data
from src.preprocessing import preprocess_data, encode_categorical, handle_missing_values
from src.feature_engineering import feature_engineering
from src.feature_selection import select_features
from src.training import train_models
from src.evaluation import evaluate_and_save_best_model

def main():
    print("Starting pipeline...")
    
    # 1. Ingestion
    print("\n--- Ingestion ---")
    file_path = "data/raw/dataset_v4.csv"
    df = load_data(file_path)
    
    # 2. Preprocessing
    print("\n--- Preprocessing ---")
    df = preprocess_data(df)
    df = handle_missing_values(df)
    
    # 3. Feature Engineering
    print("\n--- Feature Engineering ---")
    df = feature_engineering(df)
    
    # Encode categorical variables after feature engineering to ensure we have all columns ready
    # We do this before feature selection to allow RF to use them if needed, 
    # but feature selection function currently selects only numeric.
    # Let's encode here to be safe for future improvements.
    df = encode_categorical(df)
    
    # 4. Feature Selection
    print("\n--- Feature Selection ---")
    # We select top 20 features
    top_features = select_features(df, target_col='delivery_time_days', top_n=20)
    
    # 5. Training
    print("\n--- Training ---")
    models, X_test, y_test = train_models(df, target_col='delivery_time_days', feature_list=top_features)
    
    # 6. Evaluation
    print("\n--- Evaluation ---")
    evaluate_and_save_best_model(models, X_test, y_test)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
