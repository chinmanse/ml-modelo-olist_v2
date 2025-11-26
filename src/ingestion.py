import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file {file_path} is empty.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")

if __name__ == "__main__":
    # Test the ingestion
    try:
        df = load_data("data/raw/dataset_v4.csv")
        print(df.head())
    except Exception as e:
        print(e)
