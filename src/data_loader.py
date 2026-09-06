# src/data_loader
import pandas as pd
import numpy as np
import requests
import os
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.getenv("EODHD_API_KEY")
SYMBOL = "SPXL.US"
START_DATE = "2015-01-01"
DATA_DIR = "data/processed"
OUTPUT_FILE = f"{DATA_DIR}/spxl_returns.csv"

def download_data(symbol=SYMBOL, api_key=API_KEY):
    if not api_key:
        raise ValueError(
            "EODHD_API_KEY non défini — renseigne-le dans ton .env (voir .env.example)."
        )
    url = f"https://eodhistoricaldata.com/api/eod/{symbol}?api_token={api_key}&period=d&fmt=csv"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Erreur HTTP {response.status_code} lors du téléchargement.")
    
    df = pd.read_csv(StringIO(response.text), parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df

def compute_log_returns(df):
    df["log_returns"] = np.log(df["Adjusted_close"] / df["Adjusted_close"].shift(1))
    df = df.dropna()
    df = df.reset_index()
    return df

def save_to_csv(df, path=OUTPUT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=';', index=False)
    print(f" Données sauvegardées proprement dans : {path}")


def load_and_prepare_data():
    print(" Téléchargement des données...")
    df = download_data()
    print(" Calcul des rendements log...")
    df = compute_log_returns(df)
    save_to_csv(df)
    return df

# Si ce fichier est exécuté directement
if __name__ == "__main__":
    load_and_prepare_data()
