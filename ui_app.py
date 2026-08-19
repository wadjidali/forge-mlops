import streamlit as st
import requests

# URL de ton API déployée sur Render
API_URL = "https://forge-sentiment-api.onrender.com"

st.set_page_config(page_title="Forge MLOps - Sentiment Hub", page_icon="🛡️", layout="wide")

st.title("Forge MLOps - Sentiment Analysis Hub")

# Création des onglets
tab1, tab2 = st.tabs(["🧪 Prédiction en Direct", "📊 Monitoring & Data Drift"])

# --- ONGLET 1 : PRÉDICTION ---
with tab1:
    st.header("Analyse de Sentiment")
    text_input = st.text_area("Entrez le texte à analyser :", "This product is fantastic, I love it!")
    
    if st.button("Prédire le Sentiment", type="primary"):
        if text_input.strip():
            with st.spinner("Analyse en cours via l'API..."):
                try:
                    res = requests.post(f"{API_URL}/predict", json={"text": text_input})
                    if res.status_code == 200:
                        data = res.json()
                        sentiment = data.get("sentiment", "Inconnu")
                        pred = data.get("prediction", None)
                        
                        if pred == 1:
                            st.success(f"**Résultat :** {sentiment} (Classe : {pred}) 😃")
                        else:
                            st.error(f"**Résultat :** {sentiment} (Classe : {pred}) 😞")
                    else:
                        st.error(f"Erreur API ({res.status_code}) : {res.text}")
                except Exception as e:
                    st.error(f"Impossible de contacter l'API : {e}")
        else:
            st.warning("Veuillez saisir du texte.")

# --- ONGLET 2 : MONITORING ---
with tab2:
    st.header("État du Service & Dérive des Données")
    
    if st.button("🔄 Rafraîchir les métriques"):
        st.rerun()

    # 1. Vérification de l'état de l'API
    st.subheader("1. Santé du Service (Health Check)")
    try:
        health_res = requests.get(f"{API_URL}/")
        if health_res.status_code == 200:
            st.success("🟢 API en ligne et opérationnelle")
        else:
            st.warning(f"🟠 API joignable mais statut : {health_res.status_code}")
    except Exception as e:
        st.error(f"🔴 API hors ligne : {e}")

    st.divider()

    # 2. Métriques de Monitoring & Drift
    st.subheader("2. Analyse de la Dérive (Data Drift)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Valeurs issues de tes analyses de monitoring
        st.metric(label="Taux de prédictions positives (Prod)", value="67.0%", delta="Stable")
    
    with col2:
        st.metric(label="Statut du Data Drift", value="Aucune dérive", delta_color="normal")
    
    st.info("🔍 **Analyse de la distribution :** La distribution des requêtes entrantes reste conforme aux données d'entraînement baseline.")