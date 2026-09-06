# src/report_generator.py

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def generate_summary(df_alerts):
    """Génère un résumé textuel à partir du DataFrame des alertes."""
    if df_alerts.empty:
        return "Aucune alerte détectée sur la période analysée."

    worst_day = df_alerts.sort_values("predicted_volatility", ascending=False).iloc[0]
    date_max = worst_day["Date"]
    vol_max = worst_day["predicted_volatility"]
    nb_critique = (df_alerts["niveau_risque"] == "critique").sum()
    nb_elevé = (df_alerts["niveau_risque"] == "élevé").sum()

    summary = (
        f"{len(df_alerts)} jour(s) en alerte ont été détectés.\n"
        f"La volatilité la plus élevée a été observée le {date_max} avec un pic de {vol_max:.2f}%.\n"
        f"{nb_critique} jour(s) ont atteint un niveau de risque 'critique', "
        f"et {nb_elevé} un niveau 'élevé'."
    )
    return summary

def generate_pdf_report(alerts_csv, figure_path_global, figure_path_alerts, report_path, mae=None, rmse=None, asset_name="(inconnu)", sim_figure_path=None, simulation_metrics_text=None):

    """Crée un rapport PDF complet avec résumé, stats, graphiques et simulation."""

    # Charger les alertes
    df = pd.read_csv(alerts_csv, sep=';')
    df_summary = df.head(10)

    # Créer le PDF
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # -------- Page 1 : texte --------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, " Rapport SIGMA-DG1 : Alertes de Volatilité")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Généré le : {datetime.today().strftime('%d/%m/%Y %H:%M')}")

    # Nom de l’actif
    c.drawString(50, height - 85, f"Actif analysé : {asset_name}")

    if mae is not None and rmse is not None:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 100, " Évaluation du modèle :")
        c.setFont("Helvetica", 10)
        c.drawString(60, height - 115, f"MAE : {mae:.4f}")
        c.drawString(160, height - 115, f"RMSE : {rmse:.4f}")

    summary = generate_summary(df)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 145, " Résumé automatique :")
    c.setFont("Helvetica", 9)
    y = height - 160
    for line in summary.split("\n"):
        c.drawString(60, y, line)
        y -= 13

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, " Top 10 alertes détectées :")
    c.setFont("Helvetica", 9)
    y -= 30
    for _, row in df_summary.iterrows():
        line = f"{row['Date']} | σ: {row['predicted_volatility']:.2f}% | Niveau: {row['niveau_risque']}"
        c.drawString(60, y, line)
        y -= 13

    # -------- Page 2 : graphiques --------
    c.showPage()
    c.setFont("Helvetica-Bold", 12)

    if os.path.exists(figure_path_global):
        c.drawString(50, height - 50, " Volatilité GARCH sur toute la période :")
        c.drawImage(figure_path_global, x=50, y=height - 400, width=500, height=200, preserveAspectRatio=True, mask='auto')

    if os.path.exists(figure_path_alerts):
        c.drawString(50, height - 420, " Zoom sur les jours d’alerte :")
        c.drawImage(figure_path_alerts, x=50, y=height - 700, width=500, height=200, preserveAspectRatio=True, mask='auto')

    # -------- Page 3 : simulation --------
    if sim_figure_path and os.path.exists(sim_figure_path):
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 50, " Simulation de portefeuille dynamique pilotée par XGBoost :")
        c.drawImage(sim_figure_path, x=50, y=height - 500, width=500, height=300, preserveAspectRatio=True, mask='auto')

        # Ajout des métriques sous le graphique
        if simulation_metrics_text:
            y_pos = height - 430 - 250 - 20
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y_pos, " Résumé des performances :")
            c.setFont("Helvetica", 9)
            y_pos -= 15
            for line in simulation_metrics_text.strip().split("\n"):
                c.drawString(60, y_pos, line)
                y_pos -= 12

    # Finaliser le PDF
    c.save()
    print(f" Rapport PDF généré : {report_path}")
