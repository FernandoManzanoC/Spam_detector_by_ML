# Proyecto de Detección de Spam

Este repositorio contiene un proyecto de detección de spam que clasifica textos tanto en inglés como en español utilizando modelos de aprendizaje automático. La aplicación se ha desarrollado utilizando Streamlit para proporcionar una interfaz web simple y fácil de usar. El proyecto se ha realizado de manera individual para el bootcamp de Data Science de The Bridge School que finaliza en agosto de 2024.

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

- **app.py**: Este archivo contiene la aplicación de Streamlit que permite al usuario comprobar si un texto es spam o no, tanto en inglés como en español.
  
- **spam_en**: Esta carpeta contiene todo el trabajo relacionado con la detección de spam en textos en inglés.
  - **data**: Conjunto de datos utilizados para entrenar y evaluar los modelos de detección de spam en inglés.
  - **modelos_y_vectorizadores**: Modelos entrenados y vectorizadores utilizados para transformar los datos.
  - **pruebas**: Scripts y resultados de las pruebas realizadas para validar los modelos.
  - **notebooks**: Jupyter Notebooks utilizados para explorar los datos, entrenar los modelos y realizar análisis adicionales.
    - **spam_ham_dataset_modeloSVM_desbalanceo.ipynb**: Notebook principal para inglés, del cual tomamos el modelo y vectorizador para la aplicación. Este notebook incluye:
      - **Modelo SVM**: Entrenado utilizando técnicas para el desbalanceo de clases como SMOTE y ponderación de clases.
      - **Búsqueda de Hiperparámetros**: Uso de GridSearchCV para encontrar la mejor combinación de hiperparámetros.
      - **Calibración del Modelo**: Permite obtener las probabilidades de que el texto sea spam o ham.

- **spam_es**: Similar a la carpeta spam_en, pero contiene el trabajo relacionado con la detección de spam en textos en español.
  - **data**: Conjunto de datos utilizados para entrenar y evaluar los modelos de detección de spam en español.
  - **modelos_y_vectorizadores**: Modelos entrenados y vectorizadores utilizados para transformar los datos.
  - **pruebas**: Scripts y resultados de las pruebas realizadas para validar los modelos.
  - **notebooks**: Jupyter Notebooks utilizados para explorar los datos, entrenar los modelos y realizar análisis adicionales.
    - **spam_es_modelo_NB_entrenado_todotrain.ipynb**: Notebook principal para español, del cual tomamos el modelo y vectorizador entrenado con todos los datos de train para la aplicación app.py. 
      


## Cómo Ejecutar la Aplicación

1. Clona este repositorio en tu máquina local.
2. Navega al directorio del proyecto.
3. Asegúrate de tener instaladas todas las dependencias mencionadas en la sección de requisitos.
4. Ejecuta la aplicación de Streamlit con el siguiente comando:


streamlit run app.py


5. Abre tu navegador y ve a la dirección que aparece en la terminal (por lo general, es `http://localhost:8501`).

## Uso de la Aplicación

La interfaz de la aplicación te permitirá ingresar un texto y seleccionar el idioma (inglés o español). Al hacer clic en el botón de verificación, la aplicación te indicará si el texto es considerado spam o no.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit de los mismos (`git commit -am 'Añadir nueva funcionalidad'`).
4. Empuja tus cambios a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. 

## Contacto

Si tienes alguna pregunta o sugerencia, no dudes en contactar a los mantenedores del proyecto.

---

¡Gracias por usar nuestra aplicación de detección de spam!
