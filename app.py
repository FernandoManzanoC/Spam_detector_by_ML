import os
import streamlit as st
import joblib
import pickle

# Función para cargar el modelo y el vectorizador
@st.cache_resource
def load_models():
    directorio_actual = os.getcwd()
    
    # Cargar modelos
    file_path_esp = os.path.join(directorio_actual, 'spam_es', 'modelos_y_vectorizadores','modelo_naive_bayes_entrenado_completo.pkl')
    file_path_en = os.path.join(directorio_actual, 'spam_en','modelos_y_vectorizadores_en', 'calibrated_svm_model.pkl')
    
    with open(file_path_esp, 'rb') as g:
        model_es = pickle.load(g)
    
    with open(file_path_en, 'rb') as h:
        model_en = pickle.load(h)
    
    # Cargar vectorizadores
    vectorizer_file_esp = os.path.join(directorio_actual, 'spam_es', 'modelos_y_vectorizadores', 'vectorizador.pkl')
    vectorizer_file_en = os.path.join(directorio_actual, 'spam_en','modelos_y_vectorizadores_en', 'vectorizador_en.pkl')
    
    vectorizer_es = joblib.load(vectorizer_file_esp)
    vectorizer_en = joblib.load(vectorizer_file_en)
    
    return (model_es, vectorizer_es), (model_en, vectorizer_en)

(model_es, vectorizer_es), (model_en, vectorizer_en) = load_models()

# Función para predecir si el texto es spam
def predict_spam_proba(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    probas = model.predict_proba(text_vec)[0]
    spam_prob = probas[1]
    ham_prob = probas[0]
    return spam_prob, ham_prob

# Título y descripción de la aplicación
st.title("Detección de Spam en Mensajes")
st.write("""
### Bienvenido a nuestro servicio de detección de spam
Esta aplicación permite detectar si un mensaje es spam o no en inglés y español. Seleccione el idioma, ingrese el texto del mensaje y haga clic en el botón de - Detectar Spam.
""")

# Selección de idioma
language = st.selectbox("Seleccione el idioma del mensaje:", ("Español", "Inglés"))

# Entrada del usuario
st.write("#### Ingrese el texto del mensaje a evaluar:")
user_input = st.text_area("Texto del mensaje:")

# Botón de predicción
if st.button("Detectar Spam"):
    if user_input:
        if language == "Español":
            spam_prob, ham_prob = predict_spam_proba(user_input, model_es, vectorizer_es)
        else:
            spam_prob, ham_prob = predict_spam_proba(user_input, model_en, vectorizer_en)
        
        st.write(f"### Resultado: {'Spam' if spam_prob > ham_prob else 'Ham'}")
        st.write(f"Probabilidad de Spam: **{spam_prob * 100:.2f}%**")
        st.write(f"Probabilidad de No Spam (Ham): **{ham_prob * 100:.2f}%**")
    else:
        st.write("### Por favor, ingrese un texto para evaluar.")

# Footer
st.write("""
---
Desarrollado por Fernando Manzano. Utilizamos técnicas de Machine Learning para ofrecer una detección precisa de spam.
""")
