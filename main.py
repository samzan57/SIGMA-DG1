# SIGMA-DG1/main.py

from src.data_loader import load_and_prepare_data
from src.garch_model import run_training
from src.volatility_predictor import run_prediction
from src.report_generator import generate_pdf_report
from src.utils import calculate_mae_rmse
from src.simulator import load_data, engineer_features, define_target, train_model, simulate, plot, compute_metrics
from src.simulator import run_simulation
from src import simulator

def main():
    print(" Démarrage du pipeline SIGMA-DG1...\n")

    # 1. Collecte des données
    print(" 1/3 - Collecte des données")
    df_raw = load_and_prepare_data()

    # 2. Entraînement du modèle GARCH
    print("\n 2/3 - Entraînement du modèle GARCH(1,1)")
    df_garch, model = run_training()

    # Calcul MAE/RMSE
    mae, rmse = calculate_mae_rmse(df_garch["log_returns"].abs(), df_garch["predicted_volatility"])

    # 3. Prédiction et génération d'alertes
    print("\n 3/3 - Prédiction et export des alertes")
    df_final, alerts_df = run_prediction()

    print(f"\n Pipeline terminé avec succès. {len(alerts_df)} jour(s) en alerte détecté(s).")

    # 5. Simulation du portefeuille dynamique (XGBoost)
    print("\n Simulation du portefeuille avec XGBoost...")
    df_sim = simulator.load_data()
    df_sim = simulator.engineer_features(df_sim)
    df_sim = simulator.define_target(df_sim)
    df_sim = simulator.train_model(df_sim)
    df_sim = simulator.simulate(df_sim)
    simulator.plot(df_sim)
    simulation_metrics = simulator.compute_metrics(df_sim)

    # 4. Génération du rapport PDF
    print("\n Génération du rapport PDF final...")
    generate_pdf_report(
        alerts_csv="outputs/predictions/volatility_alerts.csv",
        figure_path_global="outputs/figures/sigma_dg1_volatility.png",
        figure_path_alerts="outputs/figures/sigma_dg1_volatility_alerts.png",
        report_path="outputs/volatility_report_SIGMA_DG1.pdf",
        mae=mae,
        rmse=rmse,
        asset_name="SPXL (S&P 500 3x Daily ETF)",
        sim_figure_path="outputs/figures/sigma_dg1_portfolio_simulation.png",
        simulation_metrics_text=simulation_metrics

    )
    # 5. Simulation dynamique avec XGBoost
    print("\n Simulation dynamique basée sur XGBoost...")
    df = load_data()
    df = engineer_features(df)
    df = define_target(df)
    df = train_model(df)
    df = simulate(df)
    plot(df)
    compute_metrics(df)

if __name__ == "__main__":
    main()
