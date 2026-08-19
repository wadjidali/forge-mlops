# 🛡️ Forge MLOps — Sentiment Analysis & Monitoring Hub

Un pipeline MLOps complet et automatisé pour l'entraînement, le packaging, le déploiement et le monitoring d'un modèle d'analyse de sentiment NLP.

---

## 🌟 Vue d'ensemble de l'Architecture

[ Entraînement & Tracking ] ──> MLflow (Artifact Registry)
│
[ Tests Unitaires & CI/CD ] ──> Pytest + GitHub Actions
│
[ Conteneurisation ]       ──> Docker Container
│
[ Déploiement Cloud ]      ──> Render Web Service (FastAPI)
│
[ Interface & Monitoring ] ──> Streamlit Dashboard

---

## 🚀 Fonctionnalités Clés

* **Tracking & Packaging (MLflow & Scikit-Learn) :** Entraînement d'un pipeline complet NLP (`TfidfVectorizer` + Classifieur) avec suivi des métriques et enregistrement dans le Model Registry.
* **API REST Performante (FastAPI) :** API d'inférence en production gérant la validation des requêtes avec Pydantic et l'envoi de réponses typées.
* **Intégration & Déploiement Continus (CI/CD) :** Exécution automatisée des tests unitaires via `pytest` et build/push de l'image Docker sur GitHub Actions.
* **Déploiement Conteneurisé (Docker & Render) :** Application conteneurisée et déployée sur Render avec relance automatique au push.
* **Dashboard Interactif & Monitoring (Streamlit) :** Interface utilisateur à deux onglets :
  * **🧪 Prédiction en Direct :** Test rapide d'inférence de sentiment.
  * **📊 Monitoring & Drift :** Suivi de la santé de l'API (`Health Check`) et métriques de dérive des données (`Data Drift`).

---

## 🛠️ Stack Technique

* **Langage :** Python 3.10+
* **Machine Learning & NLP :** Scikit-Learn, MLflow
* **API Framework :** FastAPI, Uvicorn, Pydantic
* **Tests Unitaires :** Pytest, HTTPX
* **Conteneurisation & CI/CD :** Docker, GitHub Actions
* **Cloud & Hosting :** Render
* **Frontend UI :** Streamlit

---

## 📁 Structure du Projet

forge-mlops/
│
├── .github/workflows/    # Pipelines CI/CD (GitHub Actions)
├── mlruns/               # Artéfacts et logs MLflow
├── app.py                # Service API REST FastAPI
├── train.py              # Script d'entraînement et d'enregistrement MLflow
├── monitoring.py         # Module de détection de dérive (Data Drift)
├── ui_app.py             # Dashboard Streamlit (Interface & Monitoring)
├── test_app.py           # Tests unitaires Pytest
├── Dockerfile            # Configuration de l'image Docker
├── requirements.txt      # Dépendances Python du projet
└── README.md             # Documentation du projet
⚙️ Installation & Lancement Local
1. Cloner le dépôt et créer un environnement virtuel
Bash
git clone [https://github.com/votre-user/forge-mlops.git](https://github.com/votre-user/forge-mlops.git)
cd forge-mlops

python -m venv venv
# Sur Windows :
.\venv\Scripts\activate
# Sur Linux/Mac :
source venv/bin/activate
2. Installer les dépendances
Bash
pip install -r requirements.txt
3. Entraîner le modèle & générer les artéfacts MLflow
Bash
python train.py
4. Lancer les tests unitaires
Bash
python -m pytest test_app.py
5. Démarrer l'API FastAPI en local
Bash
uvicorn app:app --reload
L'API sera accessible sur http://127.0.0.1:8000 (Documentation Swagger disponible sur /docs).

6. Lancer l'interface Streamlit
Bash
python -m streamlit run ui_app.py
🐳 Lancement via Docker
Bash
# Build de l'image Docker
docker build -t forge-sentiment-api .

# Exécution du conteneur
docker run -p 8000:8000 forge-sentiment-api
🔗 Endpoints API (Production)
GET / : Vérification du statut de l'API et du chargement du modèle.

POST /predict : Reçoit une chaîne de texte et renvoie la classe prédite (1 pour Positif, 0 pour Négatif).

JSON
// Exemple de Payload POST /predict
{
  "text": "This product is fantastic, I love it!"
}
📝Par Wadjid ALI
Projet réalisé dans le cadre du défi Forge MLOps.
