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
        # Cherche TOUS les fichiers nommés MLmodel peu importe la profondeur dans mlruns
        base_dir = Path(__file__).resolve().parent
        mlruns_dir = base_dir / "mlruns"
        
        found_files = list(mlruns_dir.rglob("MLmodel"))

        print(f"🔍 Fichiers trouvés : {found_files}", flush=True)

        if found_files:
            # Trier par date de modification du fichier
            found_files.sort(key=lambda p: p.stat().st_mtime)
            model_dir = str(found_files[-1].parent)
            
            print(f"📦 Chargement du modèle le plus récent depuis : {model_dir}", flush=True)
            model = mlflow.pyfunc.load_model(model_dir)
            print("✅ Modèle chargé avec succès !", flush=True)
        else:
            print("⚠️ Aucun MLmodel trouvé dans mlruns.", flush=True)
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