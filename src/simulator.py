# src/simulator.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data():
    """Charge les données prétraitées avec volatilité GARCH."""
    
    df = pd.read_csv("data/processed/spxl_volatility_garch.csv", sep=';')
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def engineer_features(df):
    """Crée les variables explicatives à partir des rendements et volatilité passée."""
    df["vol_shift"] = df["predicted_volatility"].shift(1)
    df["ret_shift"] = df["log_returns"].shift(1)
    df["vol_rolling"] = df["predicted_volatility"].rolling(window=5).mean().shift(1)
    df["ret_rolling"] = df["log_returns"].rolling(window=5).mean().shift(1)
    df.dropna(inplace=True)
    return df

def define_target(df):
    """Définit le poids optimal comme cible d'apprentissage supervisé (à automatiser)."""
    
    df["target_weight"] = df["predicted_volatility"].apply(
        lambda vol: 0.6 if vol < 5 else (0.3 if vol <= 7 else 0.2)
    )
    return df

def train_model(df):
    """Entraîne un modèle XGBoost pour prédire les poids d'allocation dynamiques."""
    X = df[["vol_shift", "ret_shift", "vol_rolling", "ret_rolling"]]
    y = df["target_weight"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)

    df["predicted_weight"] = model.predict(X_scaled)
    df["predicted_weight"] = df["predicted_weight"].clip(0.1, 0.7)  # Contraintes réalistes
    return df

def simulate(df):
    """Calcule la croissance du portefeuille dynamique et statique."""
    df["port_ml"] = (1 + df["log_returns"] * df["predicted_weight"]).cumprod()
    df["port_static"] = (1 + df["log_returns"] * 0.3).cumprod()
    return df


def plot(df):
    """Génère un graphique de la performance cumulée."""
    plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["port_ml"], label="Portefeuille ML (XGBoost)", color="darkgreen")
    plt.plot(df["Date"], df["port_static"], label="Statique (30%)", color="gray", linestyle="--")
    plt.title("Portefeuille dynamique piloté par XGBoost (SIGMA-DG1)")
    plt.xlabel("Date")
    plt.ylabel("Croissance cumulée")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # une date tous les 2 mois
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/sigma_dg1_portfolio_simulation.png", dpi=300)
    plt.close()

def compute_metrics(df):
    """Affiche les métriques de performance : valeur finale, Sharpe, drawdown."""
    def sharpe(series):
        return (series.mean() / series.std()) * np.sqrt(252)

    def max_drawdown(cum_curve):
        roll_max = cum_curve.cummax()
        drawdown = (cum_curve - roll_max) / roll_max
        return drawdown.min()

    text = "\n Métriques de performance :\n"

    for col in ["port_ml", "port_static"]:
        returns = df[col].pct_change().dropna()
        final = df[col].iloc[-1]
        sr = sharpe(returns)
        mdd = max_drawdown(df[col])
        text += f"\n{col} :\n  Final Value : {final:.4f}\n  Sharpe Ratio : {sr:.4f}\n  Max Drawdown : {mdd:.4f}\n"

    print(text)
    return text

def run_simulation():
    print(" Chargement des données...")
    df = load_data()

    print("  Création des features...")
    df = engineer_features(df)

    print(" Définition de la cible (poids)...")
    df = define_target(df)  

    print(" Entraînement du modèle XGBoost...")
    df = train_model(df)

    print(" Simulation du portefeuille...")
    df = simulate(df)

    print(" Visualisation des performances...")
    plot(df)

    metrics_text = compute_metrics(df)
    return metrics_text

if __name__ == "__main__":
    run_simulation()
