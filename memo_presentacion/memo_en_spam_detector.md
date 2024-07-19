# Spam Detection Project

### Created by Fernando Manzano Cuesta (July 2024).
### MIT License.
### GitHub User: FernandoManzanoC

In 2021, the average percentage of spam in global email traffic was 45.56%, reaching its highest proportion (46.56%) in the second quarter. [Source](https://www.kaspersky.es/resource-center/threats/spam-statistics-report-q2-2013)

During the second quarter of 2020, Spain received the highest number of spam attacks, accounting for 9.3% of the total threats, making it the global leader in spam reception. [Source](https://www.europapress.es/economia/noticia-espana-lider-mundial-recepcion-spam-segundo-trimestre-928-total-20210817160946.html)

The acceptable industry spam complaint rate is below 0.1%, equivalent to 1 complaint per 1,000 messages sent. [Source](https://help.activecampaign.com/hc/es/articles/360000150570-C%C3%B3mo-reducir-una-alta-tasa-de-quejas-de-spam#:~:text=La%20tasa%20de%20quejas%20de,por%20cada%201%2C000%20mensajes%20enviados.)

Spam is defined as any form of unsolicited communication sent in bulk. Although commonly associated with unwanted emails, spam can also appear in text messages (SMS), social media, and phone calls. Its aim is usually promotional, advertising, or, in some cases, malicious, such as phishing, where the goal is to obtain personal information from the recipient. The term "spam" originates from a sketch in the British comedy series Monty Python, where a group of Vikings repeatedly said the word "spam," symbolizing the intrusive and repetitive nature of these unsolicited messages.

Spam is a persistent and growing problem in the digital realm that affects both individual users and businesses. It not only clutters inboxes but can also contain threats like phishing, malware, and fraudulent links, compromising users' security and privacy. Additionally, the time and resources spent managing and filtering spam represent a significant cost. In this context, machine learning techniques offer a promising solution for spam detection and mitigation. By leveraging advanced algorithms that can learn and adapt from patterns in large volumes of data, it is possible to develop accurate and efficient spam filtering systems, improving security and user experience across email platforms and other digital communication forms.

This spam detection project classifies texts in both English and Spanish using machine learning models. The application was developed using Streamlit to provide a simple and user-friendly web interface. The project was carried out individually for the Data Science bootcamp at The Bridge School, concluding in August 2024.

## Data Used

### Spanish

#### Data Source
The Spanish data comes from [Hugging Face](https://huggingface.co/datasets/softecapps/spam_ham_spanish/tree/main).

#### Dataset Description
This dataset contains a total of 1000 Spanish text messages, with a label indicating whether the message is "spam" or "ham" (legitimate).

- **Message**: Contains the text of the message.
- **Label**: Indicates if the message is "spam" or "ham".

The dataset is divided into two files: `train.csv` and `test.csv`.

#### Potential Uses
- Spam filters for messaging and email services.
- Sentiment analysis in text messages.
- Detection of fraud and scams via text messages.

### English

#### Data Source
The English data comes from [Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset?resource=download).

#### Dataset Description
The spam email dataset available on Kaggle contains two main columns:

1. `text`: The body of the email, i.e., the content of the message.
2. `label`: The label indicating whether the email is spam or ham.
3. `label_num`: 0 for ham and 1 for spam.

It consists of a single file `spam_ham_dataset.csv` with 5171 entries, of which 4993 are unique values.

## Data Preprocessing

For both datasets, the following preprocessing tasks were performed:
- Removal of duplicates and repeated values.
- Removal of stopwords using NLTK.
- Removal of non-alphanumeric characters.
- Conversion to lowercase.
- Removal of extra spaces.
- Removal of prefixes like "Subject:" in messages.

## Models Used

To convert texts into numerical representations, `TfidfVectorizer` from the `scikit-learn` library in Python was used. TF-IDF (Term Frequency-Inverse Document Frequency) assigns a weight to each word in a document based on its frequency in that document and its inverse frequency in the corpus.

### Spanish

For the Spanish dataset, Naive Bayes, Random Forest, and SVM models were evaluated. Naive Bayes was ultimately chosen due to its performance:

- **Naive Bayes (NB)**: 
  - Precision: 0.89
  - Recall: 0.89
  - F1-score: 0.89
  - Confusion Matrix:
    ```
    [[94 15]
     [ 8 92]]
    ```

### English

For the English dataset, Naive Bayes and SVM models were evaluated. SVM was selected after performing GridSearchCV and optimizing parameters:

- **Support Vector Machine (SVM)**:
  - Best parameters: `{'C': 10, 'class_weight': None, 'kernel': 'rbf'}`
  - Precision: 0.98
  - Recall: 0.98
  - F1-score: 0.98
  - Confusion Matrix:
    ```
    [[719  13]
     [  7 260]]
    ```

In both cases, the final models were trained with all training data and then evaluated with the test data.

## Web Application

A Streamlit application (`app.py`) was built. The application interface allows users to input text and select the language (English or Spanish). By clicking the verification button, the application indicates whether the text is considered spam or not, and provides the probability of being spam or ham.

For this functionality, `predict_proba` was used to estimate the probability for each class, which worked directly for Naive Bayes and required calibrating the SVM model with `CalibratedClassifierCV` from `sklearn`.

## Repository Structure

- **app.py**: Contains the Streamlit application.
  
- **spam_en**: Work related to spam detection in English.
  - **data**: Datasets for training and evaluation.
  - **models_and_vectorizers**: Trained models and vectorizers.
  - **tests**: Scripts and test results.
  - **notebooks**: Notebooks for data exploration and model training.
    - [spam_ham_dataset_modeloSVM_desbalanceo.ipynb](https://github.com/FernandoManzanoC/Spam_detector_by_ML/blob/572d0c9696c889b003b82370241e3c31e57dbf53/spam_en/notebooks_entregables_en/spam_ham_dataset_modeloSVM_desbalanceo.ipynb): Main notebook for English, includes the SVM model, hyperparameter search, and model calibration.

- **spam_es**: Work related to spam detection in Spanish.
  - **data**: Datasets for training and evaluation.
  - **models_and_vectorizers**: Trained models and vectorizers.
  - **tests**: Scripts and test results.
  - **notebooks**: Notebooks for data exploration and model training.
    - [spam_es_modelo_NB_entrenado_todotrain.ipynb](https://github.com/FernandoManzanoC/Spam_detector_by_ML/blob/572d0c9696c889b003b82370241e3c31e57dbf53/spam_es/notebooks_entregables/spam_es_modelo_NB_entrenado_todotrain.ipynb): Main notebook for Spanish, includes the Naive Bayes model trained with all training data.

## Lessons Learned

- Importance of organization from the start.
- Different approaches can be used to solve the same problem.
- There is much to discover in the power and variety of existing ML and DL libraries.

## Future Improvements

- Enhance the app's content, including an explanation of its functionality.
- Apply the learned techniques to other cases such as phishing and fraudulent SMS.
- Create a Spanish spam dataset to facilitate further research on this topic.
- Improve models, especially in cases where ham is predicted but it is actually spam.
