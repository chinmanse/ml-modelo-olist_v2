import os
import joblib
import pandas as pd
from pathlib import Path

# Directorios
MODEL_DIR = Path("models")
FEATURES_PATH = Path("data/processed/feature_shortlist.csv")

def load_feature_list():
    """Carga las 20 features seleccionadas durante el entrenamiento."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"No se encontró {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    return df["feature"].tolist()

def load_best_model():
    """Detecta y carga automáticamente el modelo ganador (RF o XGBoost)."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Directorio {MODEL_DIR} no existe.")
    
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("best_model_") and fname.endswith(".pkl"):
            print(f"[INFO] Cargando modelo: {fname}")
            return joblib.load(MODEL_DIR / fname)
    
    raise FileNotFoundError("No se encontró ningún modelo 'best_model_*.pkl' en /models")

# Cargar al inicio de la API
FEATURE_LIST = load_feature_list()
MODEL = load_best_model()
