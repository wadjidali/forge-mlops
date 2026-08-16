import mlflow

# Chargement dynamique via le Model Registry et l'alias 'champion'
model_name = "Forge_Text_Classifier"
model_uri = f"models:/{model_name}@champion"

print(f"📥 Chargement du modèle '{model_name}' (Alias: champion)...")
loaded_model = mlflow.sklearn.load_model(model_uri)

print("✅ Modèle chargé avec succès depuis le registre MLflow !")