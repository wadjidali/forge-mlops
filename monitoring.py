import pandas as pd
import numpy as np

def check_data_drift():
    print("🔍 Analyse de la dérive des données (Data Drift)...")
    
    # Simulation des prédictions de production
    predictions_history = np.random.choice([0, 1], size=100, p=[0.3, 0.7])
    pos_rate = np.mean(predictions_history)
    
    print(f"📊 Taux de prédictions positives en production : {pos_rate * 100:.1f}%")
    
    # Seuil d'alerte si le modèle prédit trop d'un seul côté
    if pos_rate > 0.85 or pos_rate < 0.15:
        print("⚠️ ALERTE : Dérive potentielle détectée (Data Drift) ! Ré-entraînement recommandé.")
    else:
        print("✅ Distribution stable. Aucune dérive majeure détectée.")

if __name__ == "__main__":
    check_data_drift()