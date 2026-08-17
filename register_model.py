from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()
experiment = mlflow.get_experiment_by_name("Forge_NLP_Classification")

# Recherche du meilleur run
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"], max_results=1)
best_run_id = runs[0].info.run_id

# Enregistrement
model_uri = f"runs:/{best_run_id}/model"
result = mlflow.register_model(model_uri, "Forge_Text_Classifier")

# Alias champion
client.set_registered_model_alias("Forge_Text_Classifier", "champion", result.version)
print(f"🚀 Version {result.version} qualifiée 'champion' !")