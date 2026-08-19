import glob
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI(
    title="Forge Sentiment API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
model = None


def load_model_safely():
    global model
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            os.path.join(base_dir, "mlruns", "**", "MLmodel"),
            "./mlruns/**/MLmodel",
            "/app/mlruns/**/MLmodel"
        ]
        
        found_files = []
        for path in search_paths:
            found_files.extend(glob.glob(path, recursive=True))

        print(f"🔍 Fichiers trouvés : {found_files}", flush=True)

        if found_files:
            # 💡 Trier par date de modification pour choper le plus récent
            found_files.sort(key=os.path.getmtime)
            model_dir = os.path.dirname(found_files[-1])
            
            print(f"📦 Chargement du modèle le plus récent depuis : {model_dir}", flush=True)
            model = mlflow.pyfunc.load_model(model_dir)
            print("✅ Modèle chargé avec succès !", flush=True)
        else:
            print("⚠️ Aucun MLmodel trouvé.", flush=True)
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}", flush=True)

class TextRequest(BaseModel):
  text: str


@app.get("/")
def home():
  return {"status": "online", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: TextRequest):
    if model is None:
        raise HTTPException(
            status_code=500, detail="Modèle non chargé dans le conteneur."
        )

    try:
        # Passer le texte directement au pipeline
        preds = model.predict([request.text])
        pred_val = int(preds[0])
        
        return {
            "text": request.text,
            "prediction": pred_val,
            # Vérifie selon ton encodage y dans train.py : 
            # Si dans le dataset 1 = Positif et 0 = Négatif :
            "sentiment": "Positif" if pred_val == 1 else "Négatif",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))