# src/garch_model.py

import pandas as pd
import os
from arch import arch_model
from src.utils import calculate_mae_rmse

# Chemins des fichiers
INPUT_FILE = "data/processed/spxl_returns.csv"
OUTPUT_FILE = "data/processed/spxl_volatility_garch.csv"

# Chargement des rendements
def load_returns(path=INPUT_FILE):
    df = pd.read_csv(path, sep=';')
    return df

# Entraînement du modèle GARCH
def train_garch_model(returns):
    model = arch_model(returns * 100, vol='GARCH', p=1, q=1)
    model_fit = model.fit(disp='off')
    return model_fit

# Ajout des prédictions au DataFrame
def add_volatility_to_df(df, model_fit):
    df["predicted_volatility"] = model_fit.conditional_volatility
    return df

# Sauvegarde propre du fichier
def save_results(df, path=OUTPUT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=';', index=False)
    print(f" Résultats sauvegardés dans : {path}")

# Pipeline principal
def run_training():
    print(" Chargement des rendements...")
    df = load_returns()
    
    print(" Entraînement du modèle GARCH(1,1)...")
    model_fit = train_garch_model(df["log_returns"])
    
    print(" Ajout de la volatilité prédite...")
    df = add_volatility_to_df(df, model_fit)
    
    print(" Évaluation du modèle...")
    mae, rmse = calculate_mae_rmse(df["log_returns"].abs(), df["predicted_volatility"])
    print(f" MAE : {mae:.4f} | RMSE : {rmse:.4f}")
    
    save_results(df)
    return df, model_fit

# Exécution directe
if __name__ == "__main__":
    df, model_fit = run_training()
