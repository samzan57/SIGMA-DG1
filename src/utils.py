# src/utils.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def calculate_mae_rmse(y_true, y_pred):
    """Calcule MAE et RMSE entre deux séries."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def classify_risk(vol, seuils=(3, 6)):
    """
    Attribue un niveau de risque basé sur le niveau de volatilité.
    seuils = (seuil_moderé, seuil_élevé)
    """
    if vol < seuils[0]:
        return "modéré"
    elif vol < seuils[1]:
        return "élevé"
    else:
        return "critique"

def apply_risk_level(df, col_vol="predicted_volatility"):
    """Ajoute une colonne 'niveau_risque' en fonction de la volatilité prédite."""
    df["niveau_risque"] = df[col_vol].apply(classify_risk)
    return df

def export_csv(df, path, sep=";"):
    """Sauvegarde propre d’un DataFrame au format CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=sep, index=False)
    print(f" Exporté dans : {path}")
