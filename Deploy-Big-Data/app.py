import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan TF-IDF
model = joblib.load("decision_tree_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Judul halaman
st.title("Klasifikasi Sentimen Ulasan Game Honkai: Star Rail")

# Input teks
user_input = st.text_area("Masukkan ulasan Anda di sini:")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Ulasan tidak boleh kosong.")
    else:
        # TF-IDF transform
        user_tfidf = tfidf.transform([user_input])
        prediction = model.predict(user_tfidf)
        label = "Positif" if prediction[0] == 1 else "Negatif"
        st.success(f"Prediksi Sentimen: **{label}**")
