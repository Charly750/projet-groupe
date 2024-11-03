import streamlit as st
import joblib
import numpy as np
import sqlite3
import pandas as pd
# Chargement du modèle pré-entraîné et de l'erreur MAE
model = joblib.load('RFR_model.pkl')
mae = 140.15  # Remplacez cette valeur par le MAE obtenu lors de l'évaluation du modèle
conn = sqlite3.connect("annonces.db")
#je récupère les données
data = pd.read_sql_query("SELECT * FROM annonces", conn)
#je ferme la connexion
conn.close()
st.title("Prédiction de Prix de Loyer avec Marge d'Erreur")

st.write("Entrez les informations suivantes pour prédire le prix de loyer :")

dataForRFR = data[data["price"] < 3300]
#on utilise la surface, et le nombre de piece maximum dans dataForRFR pour définir les valeurs max
max_surface = dataForRFR["surface"].max()
max_pieces = dataForRFR["pieces"].max()
# Formulaire de saisie des données
surface = st.number_input("Surface (en m²)", min_value=10.0, max_value=max_surface, step=0.1)
location = st.number_input("Arrondissement", min_value=1, max_value=20, step=1)
pieces = st.number_input("Nombre de pièces", min_value=1, max_value=max_pieces, step=1)

# Bouton de prédiction
if st.button("Prédire le Prix"):
    # Préparation des données pour la prédiction
    input_data = np.array([[surface, location, pieces]])
    prediction = model.predict(input_data)[0]
    
    # Calcul de la marge d'erreur
    lower_bound = prediction - mae
    upper_bound = prediction + mae
    
    # Affichage du résultat
    st.write(f"Prix prédit : {prediction:.2f} €")
    st.write(f"Intervalle de confiance : {lower_bound:.2f} € - {upper_bound:.2f} €")
