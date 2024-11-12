import streamlit as st
import pickle
import numpy as np

# Tahmin fonksiyonu
def text_pred(full_text):
    # Model ve vektörizer'i her tahminde yeniden yükleme
    with open('laboratuvar_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('laboratuvar_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Giriş verisini vektörize etme
    input_data = vectorizer.transform([full_text]).toarray()

    # Tahmin yapma
    prediction = model.predict(input_data)
    return float(prediction[0])

# Streamlit arayüzü
st.title("Metin Tabanlı Tahmin")
st.write("Veriyi Gir")

full_text = st.text_area('Metin Girin:', value='', height=100)

if st.button('Tahmin Et'):
    prediction = text_pred(full_text)
    st.write(f'Tahmin edilen skor: {prediction:.2f}')
