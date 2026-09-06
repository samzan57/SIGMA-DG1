import pandas as pd
import numpy as np
from arch import arch_model
import os
from src.utils import apply_risk_level, export_csv

# Fichiers et paramètres
INPUT_FILE = "data/processed/spxl_returns.csv"
ALERTS_OUTPUT = "outputs/predictions/volatility_alerts.csv"
SEUIL_VOL = 5.0  # seuil d'alerte en %

# Chargement des rendements
def load_returns(path=INPUT_FILE):
    df = pd.read_csv(path, sep=';')
    return df

# Prédiction GARCH(1,1)
def predict_volatility(df):
    model = arch_model(df["log_returns"] * 100, vol='GARCH', p=1, q=1)
    model_fit = model.fit(disp='off')
    df["predicted_volatility"] = model_fit.conditional_volatility
    return df

# Génération des alertes
def generate_alerts(df, seuil=SEUIL_VOL):
    df["alerte"] = df["predicted_volatility"] > seuil
    alerts_df = df[df["alerte"]].copy()
    return alerts_df

# Pipeline principal
def run_prediction():
    print(" Chargement des rendements...")
    df = load_returns()
    
    print(" Prédiction de la volatilité...")
    df = predict_volatility(df)
    
    print(f" Génération des alertes (vol > {SEUIL_VOL}%)...")
    alerts_df = generate_alerts(df)
    
    print(" Attribution du niveau de risque...")
    alerts_df = apply_risk_level(alerts_df)
    
    print(" Export final des alertes...")
    export_csv(alerts_df, ALERTS_OUTPUT)
    
    return df, alerts_df

# Exécution directe
if __name__ == "__main__":
    df, alerts_df = run_prediction()
