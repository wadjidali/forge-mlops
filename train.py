import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Set experiment name
mlflow.set_experiment("Forge_NLP_Classification")

def load_data():
    # Dataset de classification de sentiment (avis produits), généré à partir de
    # modèles de phrases + vocabulaire de sentiment. Approche volontaire :
    # - phrases uniques (pas de duplication) pour éviter toute fuite train/test
    # - vocabulaire de sentiment qui se répète across phrases (comme un vrai corpus),
    #   sinon un modèle bag-of-words n'a aucun signal à apprendre sur un petit dataset
    import itertools
    import random
    random.seed(42)

    templates_pos = [
        "This {noun} is {adj}.",
        "I think this {noun} is really {adj}.",
        "Such a {adj} {noun}, I am impressed.",
        "The {noun} works great, {adj} overall.",
        "Absolutely {adj} {noun}, would buy again.",
        "What a {adj} {noun}, highly recommended.",
    ]
    templates_neg = [
        "This {noun} is {adj}.",
        "I think this {noun} is really {adj}.",
        "Such a {adj} {noun}, I am disappointed.",
        "The {noun} barely works, {adj} overall.",
        "Absolutely {adj} {noun}, would not buy again.",
        "What a {adj} {noun}, not recommended.",
    ]
    nouns = ["product", "item", "gadget", "device", "purchase", "tool", "accessory", "headset", "laptop", "service"]
    adj_pos = ["great", "excellent", "amazing", "fantastic", "wonderful", "superb", "impressive", "outstanding"]
    adj_neg = ["terrible", "awful", "poor", "disappointing", "horrible", "subpar", "mediocre", "frustrating"]

    def generate(templates, adjs, n):
        combos = list(itertools.product(templates, nouns, adjs))
        random.shuffle(combos)
        seen, out = set(), []
        for t, noun, adj in combos:
            s = t.format(noun=noun, adj=adj)
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) == n:
                break
        return out

    positive = generate(templates_pos, adj_pos, 60)
    negative = generate(templates_neg, adj_neg, 60)

    df = pd.DataFrame({
        "text": positive + negative,
        "label": [1] * len(positive) + [0] * len(negative)
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # mélange


def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_experiment(model_type="random_forest", params=None):
    if params is None:
        params = {}

    df = load_data()
    # stratify=df['label'] garantit un ratio positif/négatif identique dans train et test
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
    )

    # Preprocessing TF-IDF
    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    run_name = f"Run_{model_type.upper()}"

    with mlflow.start_run(run_name=run_name):
        # Enregistrement des paramètres TF-IDF
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", 200)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        if model_type == "random_forest":
            n_estimators = params.get("n_estimators", 100)
            max_depth = params.get("max_depth", 5)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train_vec, y_train)

        elif model_type == "xgboost":
            n_estimators = params.get("n_estimators", 100)
            learning_rate = params.get("learning_rate", 0.1)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)

            model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            model.fit(X_train_vec, y_train)

        # Prédictions et métriques
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # Enregistrement des métriques
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Matrice de confusion en tant qu'artefact
        cm_filename = f"cm_{model_type}.png"
        save_confusion_matrix(y_test, y_pred, cm_filename)
        mlflow.log_artifact(cm_filename)
        if os.path.exists(cm_filename):
            os.remove(cm_filename)

        # Sauvegarde du modèle
        if model_type == "random_forest":
            mlflow.sklearn.log_model(model, "model")
        else:
            mlflow.xgboost.log_model(model, "model")

        print(f"[{model_type.upper()}] Run termine. Accuracy: {acc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    # Run 1: Random Forest
    run_experiment("random_forest", {"n_estimators": 50, "max_depth": 3})

    # Run 2: XGBoost
    run_experiment("xgboost", {"n_estimators": 100, "learning_rate": 0.05})