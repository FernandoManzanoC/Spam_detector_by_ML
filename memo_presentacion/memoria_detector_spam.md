# Proyecto de Detección de Spam

### Elaborado por Fernando Manzano Cuesta (Julio 2024).
### Licencia MIT.
### Usuario Gihug: FernandoManzanoC

En 2021, el porcentaje promedio de spam en el tráfico mundial de correo electrónico fue del 45,56%, alcanzando su mayor proporción (46,56%) en el segundo trimestre. [Fuente](https://www.kaspersky.es/resource-center/threats/spam-statistics-report-q2-2013)

Durante el segundo trimestre de 2020, España recibió el mayor número de ataques de spam, con un 9,3% del total de estas amenazas, convirtiéndose en el líder mundial en recepción de spam. [Fuente](https://www.europapress.es/economia/noticia-espana-lider-mundial-recepcion-spam-segundo-trimestre-928-total-20210817160946.html)

La tasa de quejas de spam aceptable en la industria es inferior al 0.1%, equivalente a 1 queja por cada 1,000 mensajes enviados. [Fuente](https://help.activecampaign.com/hc/es/articles/360000150570-C%C3%B3mo-reducir-una-alta-tasa-de-quejas-de-spam#:~:text=La%20tasa%20de%20quejas%20de,por%20cada%201%2C000%20mensajes%20enviados.)

El spam se define como cualquier forma de comunicación no solicitada enviada de manera masiva. Aunque comúnmente asociado con correos electrónicos no deseados, el spam también puede manifestarse en mensajes de texto (SMS), redes sociales y llamadas telefónicas. Su objetivo suele ser promocional, publicitario o, en algunos casos, malicioso, como en el phishing, donde se busca obtener información personal del destinatario. El término "spam" proviene de un sketch de la serie de comedia británica Monty Python, donde un grupo de vikingos repetía insistentemente la palabra "spam", simbolizando la naturaleza intrusiva y repetitiva de estos mensajes no solicitados.

El spam es un problema persistente y creciente en el ámbito digital que afecta tanto a usuarios individuales como a empresas. No solo satura las bandejas de entrada, sino que también puede contener amenazas como phishing, malware y enlaces fraudulentos, comprometiendo la seguridad y privacidad de los usuarios. Además, el tiempo y los recursos dedicados a gestionar y filtrar el spam representan un costo significativo. En este contexto, las técnicas de machine learning ofrecen una solución prometedora para la detección y mitigación del spam. Al aprovechar algoritmos avanzados que pueden aprender y adaptarse a partir de patrones en grandes volúmenes de datos, es posible desarrollar sistemas de filtrado de spam que sean precisos y eficientes, mejorando la seguridad y la experiencia del usuario en plataformas de correo electrónico y otras formas de comunicación digital.

Este proyecto de detección de spam clasifica textos tanto en inglés como en español utilizando modelos de aprendizaje automático. La aplicación se ha desarrollado utilizando Streamlit para proporcionar una interfaz web simple y fácil de usar. El proyecto se ha realizado de manera individual para el bootcamp de Data Science de The Bridge School, que finaliza en agosto de 2024.

## Datos Utilizados

### Español

#### Origen de los Datos
Los datos en español provienen de [Hugging Face](https://huggingface.co/datasets/softecapps/spam_ham_spanish/tree/main).

#### Descripción del Dataset
Este dataset contiene un total de 1000 mensajes de texto en español, con una etiqueta que indica si el mensaje es "spam" o "ham" (legítimo).

- **Mensaje**: Contiene el texto del mensaje.
- **Etiqueta**: Indica si el mensaje es "spam" o "ham".

El dataset está dividido en dos archivos: `train.csv` y `test.csv`.

#### Potenciales Usos
- Filtros de spam para servicios de mensajería y email.
- Análisis de sentimiento en mensajes de texto.
- Detección de fraude y estafas a través de mensajes de texto.

### Inglés

#### Origen de los Datos
Los datos en inglés provienen de [Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset?resource=download).

#### Descripción del Dataset
El dataset de correos spam disponible en Kaggle contiene dos columnas principales:

1. `text`: El cuerpo del correo electrónico, es decir, el contenido del mensaje.
2. `label`: La etiqueta que indica si el correo es spam o ham.
3. `label_num`: 0 para ham y 1 para spam.

Consta de un único archivo `spam_ham_dataset.csv` con 5171 entradas, de las cuales 4993 son valores únicos.

## Preprocesamiento de Datos

Para ambos datasets, se realizaron las siguientes tareas de preprocesamiento:
- Eliminación de duplicados y valores repetidos.
- Eliminación de palabras vacías utilizando stopwords de NLTK.
- Eliminación de caracteres no alfanuméricos.
- Conversión a minúsculas.
- Eliminación de espacios extra.
- Eliminación de prefijos como "Subject:" en los mensajes.

## Modelos Utilizados

Para convertir los textos en una representación numérica, se utilizó `TfidfVectorizer` de la biblioteca `scikit-learn` en Python. TF-IDF (Term Frequency-Inverse Document Frequency) asigna un peso a cada palabra en un documento, basado en su frecuencia en ese documento y su frecuencia inversa en el conjunto de documentos.

### Español

Para el dataset en español, se evaluaron los modelos Naive Bayes, Random Forest y SVM. Finalmente, se seleccionó Naive Bayes debido a su rendimiento:

- **Naive Bayes (NB)**: 
  - Precisión: 0.89
  - Recall: 0.89
  - F1-score: 0.89
  - Matriz de Confusión:
    ```
    [[94 15]
     [ 8 92]]
    ```

### Inglés

Para el dataset en inglés, se evaluaron los modelos Naive Bayes y SVM. Se seleccionó SVM tras realizar un GridSearchCV y optimizar parámetros:

- **Support Vector Machine (SVM)**:
  - Mejores parámetros: `{'C': 10, 'class_weight': None, 'kernel': 'rbf'}`
  - Precisión: 0.98
  - Recall: 0.98
  - F1-score: 0.98
  - Matriz de Confusión:
    ```
    [[719  13]
     [  7 260]]
    ```

En ambos casos, los modelos finales fueron entrenados con todos los datos de entrenamiento y luego evaluados con los datos de prueba.

## Aplicación Web

Se construyó una aplicación en Streamlit (`app.py`). La interfaz de la aplicación permite ingresar un texto y seleccionar el idioma (inglés o español). Al hacer clic en el botón de verificación, la aplicación indica si el texto es considerado spam o no, y proporciona la probabilidad de que sea spam o ham.

Para esta funcionalidad, se utilizó la función `predict_proba`, que devuelve la probabilidad estimada de cada clase para las muestras de entrada. Esto funcionó directamente para Naive Bayes y requirió calibrar el modelo SVM con `CalibratedClassifierCV` de `sklearn`.

## Estructura del Repositorio

- **app.py**: Contiene la aplicación de Streamlit.
  
- **spam_en**: Trabajo relacionado con la detección de spam en inglés.
  - **data**: Conjunto de datos para entrenamiento y evaluación.
  - **modelos_y_vectorizadores**: Modelos entrenados y vectorizadores.
  - **pruebas**: Scripts y resultados de pruebas.
  - **notebooks**: Notebooks para exploración de datos y entrenamiento de modelos.
    - [spam_ham_dataset_modeloSVM_desbalanceo.ipynb](https://github.com/FernandoManzanoC/Spam_detector_by_ML/blob/572d0c9696c889b003b82370241e3c31e57dbf53/spam_en/notebooks_entregables_en/spam_ham_dataset_modeloSVM_desbalanceo.ipynb)
: Notebook principal para inglés, incluye el modelo SVM, búsqueda de hiperparámetros y calibración del modelo.

- **spam_es**: Trabajo relacionado con la detección de spam en español.
  - **data**: Conjunto de datos para entrenamiento y evaluación.
  - **modelos_y_vectorizadores**: Modelos entrenados y vectorizadores.
  - **pruebas**: Scripts y resultados de pruebas.
  - **notebooks**: Notebooks para exploración de datos y entrenamiento de modelos.
    - [spam_es_modelo_NB_entrenado_todotrain.ipynb](https://github.com/FernandoManzanoC/Spam_detector_by_ML/blob/572d0c9696c889b003b82370241e3c31e57dbf53/spam_es/notebooks_entregables/spam_es_modelo_NB_entrenado_todotrain.ipynb)
: Notebook principal para español, incluye el modelo Naive Bayes entrenado con todos los datos de entrenamiento.

## Lecciones Aprendidas

- Importancia del orden desde el primer momento.
- Existen distintos enfoques de solución para un mismo problema.
- Hay mucho que descubrir en la potencia y variedad de las bibliotecas existentes de ML y DL.

## Futuras Mejoras

- Mejorar el contenido de la app, incluyendo una explicación de su funcionamiento.
- Aplicación de lo aprendido a otros casos como phishing y SMS fraudulentos.
- Creación de un dataset de spam en español para facilitar más trabajos sobre este tema.
- Mejora de modelos, especialmente en los casos en que se predice ham y es spam.


