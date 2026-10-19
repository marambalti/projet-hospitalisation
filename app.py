import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Durée d'hospitalisation", page_icon="🏥")
st.title("🏥 Prédiction de la durée d'hospitalisation")
st.write("Séjour **court** (3 jours ou moins) ou **long** (plus de 3 jours) ?")


@st.cache_resource
def charger():
    modele = joblib.load("modele_final.joblib")
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return modele, meta


modele, meta = charger()

st.subheader("Informations du patient")
valeurs = {}
col1, col2 = st.columns(2)

for i, (nom, info) in enumerate(meta["numeriques"].items()):
    with (col1 if i % 2 == 0 else col2):
        valeurs[nom] = st.number_input(
            nom,
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["mediane"]),
        )

for i, (nom, options) in enumerate(meta["categorielles"].items()):
    with (col1 if i % 2 == 0 else col2):
        valeurs[nom] = st.selectbox(nom, options)

if st.button("Prédire", type="primary"):
    ligne = pd.DataFrame([valeurs])[meta["ordre_colonnes"]]
    proba_longue = modele.predict_proba(ligne)[0][1]
    if proba_longue >= 0.5:
        st.error(f"Séjour probablement LONG (> 3 jours) — probabilité : {proba_longue:.0%}")
    else:
        st.success(f"Séjour probablement COURT (≤ 3 jours) — probabilité de séjour long : {proba_longue:.0%}")
    st.caption("Outil pédagogique : ne remplace pas un avis médical.")
