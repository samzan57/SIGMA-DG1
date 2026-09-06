"""
Unit tests for the risk-classification and alerting logic
(src/utils.py, src/volatility_predictor.py).

Run with:
    pytest tests/
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import classify_risk, apply_risk_level, calculate_mae_rmse
from src.volatility_predictor import generate_alerts


def test_classify_risk_boundaries():
    assert classify_risk(2.9) == "modéré"
    assert classify_risk(3.0) == "élevé"     # borne basse incluse dans "élevé"
    assert classify_risk(5.9) == "élevé"
    assert classify_risk(6.0) == "critique"  # borne haute bascule en "critique"
    assert classify_risk(10.0) == "critique"


def test_apply_risk_level_adds_expected_column():
    df = pd.DataFrame({"predicted_volatility": [1.0, 4.0, 8.0]})

    result = apply_risk_level(df)

    assert list(result["niveau_risque"]) == ["modéré", "élevé", "critique"]


def test_generate_alerts_flags_only_days_above_threshold():
    df = pd.DataFrame({"predicted_volatility": [2.0, 5.5, 4.9, 7.0]})

    alerts = generate_alerts(df, seuil=5.0)

    assert len(alerts) == 2
    assert list(alerts["predicted_volatility"]) == [5.5, 7.0]


def test_calculate_mae_rmse_matches_known_values():
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 5.0])  # one error of 2

    mae, rmse = calculate_mae_rmse(y_true, y_pred)

    assert mae == 2 / 3
    assert round(rmse, 6) == round((4 / 3) ** 0.5, 6)
