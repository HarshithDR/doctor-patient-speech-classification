# Doctor-Patient Audio Diarization Using ML Models

## Overview
This project aims to develop machine learning models to classify doctor-patient conversations and automate the generation of medical reports. By leveraging natural language processing (NLP) and various machine learning algorithms, we aim to streamline the documentation process in healthcare, providing efficient and objective insights from medical conversations.

## Team Members
- Sachin Shivanna
- Uday Venkatesha
- Rajashekar Allam
- Nishchal Gante Ravish
- Harshith Deshalli Ravi

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Setting](#problem-setting)
3. [Data Description](#data-description)
4. [Data Preprocessing](#data-preprocessing)
5. [Models Used and Analysis](#models-used-and-analysis)
6. [Model Comparison](#model-comparison)
7. [Conclusion](#conclusion)
8. [Challenges](#challenges)
9. [Future Work](#future-work)

## Introduction
The use of technology in generating medical reports can significantly reduce the time physicians spend on documentation. According to the National Institutes of Health, physicians spend around 35% of their time documenting patient data. This project aims to automate this process, providing a centralized and efficient way to handle patient health records.

## Problem Setting
### Objective
- Develop models to classify doctor-patient conversations.
- Automate the generation of medical reports.

Healthcare data is insightful, transformative, and invaluable. Unlike traditional methods, our model automates the analysis, providing efficient and objective insights for healthcare, which can be used as a centralized patient health record.

## Data Description
### Source
The dataset comprises statements extracted from interactions between doctors and patients, representing a diverse range of medical conversations. The data is sourced from Kaggle: [Diagnoise Me Dataset](https://www.kaggle.com/datasets/dsxavier/diagnoise-me).

### Features and Labels
- **Features:** The textual content of each statement serves as the primary feature for analysis, capturing the language patterns and nuances used by doctors and patients.
- **Labels:** Each statement is labeled as either "doctor" or "patient" to facilitate supervised learning.

## Data Preprocessing
### Actions Taken
- Removed unnecessary fields such as 'id' and 'Description' from the dataset to simplify the data structure and focus on relevant information.
- Stored the cleaned data into a new file with a total of 514,842 datapoints.

## Models Used and Analysis
We analyzed the following models:
- Logistic Regression
- Support Vector Machines (SVM)
- Naive Bayes
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Random Forest
- Feed Forward Neural Networks

### Model Performance
- **Logistic Regression:** Train Accuracy: 99.8%, Test Accuracy: 99.52%
- **Support Vector Machines:** Train Accuracy: 99.9%, Test Accuracy: 99.47%
- **Naive Bayes:** Train Accuracy: 99.06%, Test Accuracy: 98.97%
- **Random Forest:** Train Accuracy: 99.9%, Test Accuracy: 97.81%
- **RNN:** Train Accuracy: 99.06%, Test Accuracy: 98.97%
- **LSTM:** Test Accuracy: 97.81%
- **Feed Forward Neural Networks:** Train Accuracy: 99.9%, Test Accuracy: 97.81%

## Model Comparison
### Accuracy Comparison
Logistic Regression outperformed other models with the highest test accuracy of 99.52%.

### Model Size and Training Time
- **Logistic Regression:** Model Size: 1.91 MB, Training Time: 65 seconds

## Conclusion
With the above analysis, we conclude that Logistic Regression is the best fit for text classification in this context, achieving the highest accuracy and reasonable training time and model size.

## Challenges
- **Variability in Language:** Medical conversations involve varied terminology, levels of formality, and context-dependent language.
- **Data Imbalance:** Unequal distribution of statements from doctors and patients can pose challenges.
- **Data Privacy:** Maintaining sensitive health data requires additional legal and security measures.

## Future Work
- Integrating the model into IoT devices and applications for practical usage.
- Using the model output as a patient’s health record for future reference.
