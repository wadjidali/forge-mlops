import mlflow
from mlflow.tracking import MlflowClient

# 1. Connexion au client MLflow
client = MlflowClient()
experiment_name = "Forge_NLP_Classification"
experiment = client.get_experiment_by_name(experiment_name)

if not experiment:
    raise ValueError(f"L'expérience '{experiment_name}' n'a pas été trouvée.")

# 2. Recherche du meilleur Run basé sur l'accuracy
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_accuracy = best_run.data.metrics.get('accuracy', 0)
model_type = best_run.data.params.get('model_type', 'inconnu')

print(f"🏆 Meilleur Run trouvé : ID {best_run_id} ({model_type}) avec Accuracy = {best_accuracy}")

# 3. Enregistrement dans le Model Registry
model_name = "Forge_Text_Classifier"
model_uri = f"runs:/{best_run_id}/model"

result = mlflow.register_model(model_uri, model_name)
print(f"✅ Modèle enregistré sous le nom '{model_name}' (Version {result.version})")

# 4. Attribution de l'alias 'champion'
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=result.version
)
print(f"🚀 Version {result.version} qualifiée avec l'alias 'champion' !")