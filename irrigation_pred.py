import streamlit as st
import pandas as pd
import joblib
import os

# chemin absolu relatif au fichier .py
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "rf_irrigation_model.joblib")

# --- Titre ---
st.title("Prédiction du Besoin en :red[Irrigation]")

st.markdown("#### :red[Projet fil conducteur]")
st.markdown("Auteur : :red[**Osée AKOUEHOU, Data Analyst | Business Intelligence | Data Scientist**]")
st.markdown("Contact: +229 01 54 03 10 10 | Email: oseejoh09@gmail.com")

# --- Charger le modèle ---
pipe_rf = joblib.load("rf_irrigation_model.joblib")

# --- Formulaire pour saisir les variables ---
st.header("Entrez les caractéristiques de la parcelle")

# Exemple des variables (tu peux adapter toutes tes colonnes)
Soil_Type = st.selectbox("Type de sol", ["Sandy", "Clay", "Loam"])
Soil_Moisture = st.number_input("Humidité du sol (%)", min_value=0.0, max_value=100.0, value=25.0)
Rainfall_mm = st.number_input("Précipitations (mm)", min_value=0.0, value=20.0)
Crop_Type = st.selectbox("Type de culture", ["Wheat", "Corn", "Rice"])
Crop_Growth_Stage = st.number_input("Stade de croissance", min_value=1, max_value=5, value=2)
Season = st.selectbox("Saison", ["Zaid", "Kharif", "Rabi"])
Field_Area_hectare = st.number_input("Superficie (hectares)", min_value=0.0, value=1.0)
Mulching_Used = st.selectbox("Paillage utilisé ?", [0, 1])
Previous_Irrigation_mm = st.number_input("Irrigation précédente (mm)", min_value=0.0, value=10.0)
Region = st.selectbox("Région", ["North", "South", "East", "West"])

# --- Créer DataFrame pour le modèle ---
X_new = pd.DataFrame([{
    'Soil_Type': Soil_Type,
    'Soil_Moisture': Soil_Moisture,
    'Rainfall_mm': Rainfall_mm,
    'Crop_Type': Crop_Type,
    'Crop_Growth_Stage': Crop_Growth_Stage,
    'Season': Season,
    'Field_Area_hectare': Field_Area_hectare,
    'Mulching_Used': Mulching_Used,
    'Previous_Irrigation_mm': Previous_Irrigation_mm,
    'Region': Region
}])

# --- Bouton de prédiction ---
if st.button("Prédire le besoin en irrigation"):
    y_pred = pipe_rf.predict(X_new)[0]
    y_prob = pipe_rf.predict_proba(X_new)[0]

    st.subheader("Résultat :")
    st.write(f"**Besoin en irrigation prédit : {y_pred}**")
    
    st.subheader("Probabilités :")
    st.write(f"Low : {y_prob[0]:.2f}")
    st.write(f"Medium : {y_prob[1]:.2f}")
    st.write(f"High : {y_prob[2]:.2f}")