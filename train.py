import itertools
import os
import random
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Config MLflow
mlflow.set_experiment("Forge_NLP_Classification")


def load_data():
    random.seed(42)

    pos_sentences = [
        "This product is fantastic, I love it!",
        "Great item, highly recommended and wonderful quality.",
        "Amazing purchase, works great and superb performance.",
        "I think this gadget is really excellent and impressive.",
        "Absolutely wonderful tool, would buy again without hesitation.",
        "What a fantastic device, outstanding service!",
        "Superb experience, awesome product and great features.",
        "Very happy with this item, fantastic quality and value.",
    ]

    neg_sentences = [
        "This gadget is terrible and completely useless.",
        "Awful product, very disappointing and poor quality.",
        "I hate this item, it barely works and is terrible.",
        "Horrible experience, subpar performance and frustrating.",
        "Disappointing purchase, would not buy again.",
        "What a terrible tool, not recommended at all.",
        "Subpar quality, mediocre device and poor service.",
        "Very unhappy with this purchase, awful and broken.",
    ]

    pos_data = pos_sentences * 15
    neg_data = neg_sentences * 15

    df = pd.DataFrame({
        "text": pos_data + neg_data,
        "label": [1] * len(pos_data) + [0] * len(neg_data),
    })

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def run_experiment(params=None):
    if params is None:
        params = {}

    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    with mlflow.start_run(run_name="Run_RANDOM_FOREST"):
        vectorizer = TfidfVectorizer(
            max_features=300, ngram_range=(1, 2), stop_words="english"
        )

        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            random_state=42,
        )

        pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        # Enregistrement propre du Pipeline Sklearn complet
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model"
        )
        print(f"[RANDOM_FOREST] Run terminé. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_experiment({"n_estimators": 100, "max_depth": 5})