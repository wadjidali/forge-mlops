import streamlit as st
import requests

st.set_page_config(page_title="Forge - Sentiment Analyzer", page_icon="⚡")

st.title("⚡ Forge MLOps — Analyseur de Sentiments")
st.write("Interface connectée à l'API FastAPI déployée sur Render.")

# URL de ton API Render
API_URL = "https://forge-sentiment-api.onrender.com/predict"

# Champ de saisie
user_text = st.text_area("Entrez un texte à analyser :", "This product is fantastic, I love it!")

if st.button("Analyser le sentiment"):
    if user_text.strip():
        with st.spinner("Analyse en cours via l'API..."):
            try:
                response = requests.post(API_URL, json={"text": user_text})
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data.get("sentiment", "Inconnu")
                    
                    if sentiment == "Positif":
                        st.success(f"Resultat : **{sentiment}** 😃")
                    else:
                        st.error(f"Resultat : **{sentiment}** 🙁")
                else:
                    st.warning(f"Erreur API ({response.status_code})")
            except Exception as e:
                st.error(f"Impossible de contacter l'API : {e}")
    else:
        st.info("Veuillez saisir du texte.")