# Spam Detection using Machine Learning

This project is a text classification system that detects whether a message is **spam** or **ham (not spam)**. It uses classical **machine learning models** such as Naive Bayes, Random Forest, SVM, and ensemble methods to classify SMS messages.

##Dataset
- SMS Spam Collection dataset (from UCI repository or CEAS)
- Total messages: 5169 (4516 ham, 653 spam)

## Models Used
- **Naive Bayes (Multinomial, Gaussian, Bernoulli)**
- **Logistic Regression**
- **Support Vector Machine (SVC)**
- **Decision Tree, Random Forest**
- **Ensemble models**: AdaBoost, Bagging, Extra Trees, Gradient Boosting
- **Voting Classifier** (best model using NB + RF + ETC)

## NLP Techniques Applied
- Tokenization (using NLTK)
- Lowercasing
- Stop word & punctuation removal
- Stemming (PorterStemmer)
- TF-IDF and CountVectorizer for text vectorization

## Evaluation Metrics
- Accuracy
- Precision (focused due to class imbalance)
- Confusion Matrix
- Heatmap of model performance

## Best Performing Model
- **VotingClassifier (NB + Random Forest + Extra Trees)**
- **Precision:** 1.0  
- **Accuracy:** ~97%

## Libraries Used
- Python
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- nltk
- wordcloud

## Project Highlights
- Detailed preprocessing pipeline
- Word cloud for spam vs ham
- Keyword frequency analysis
- Model comparison chart with precision + accuracy

---
# Machine Failure Prediction using Logistic Regression

This project builds a **binary classification model** to predict whether a machine will fail based on various sensor inputs and environmental factors.

## Dataset
- Custom sensor dataset with 944 rows and 10 columns
- Features include: footfall, temperature, air quality, usage stats, VOC, internal pressure, etc.
- Target: `fail` (0 = No failure, 1 = Failure)

## Problem Type
- Supervised learning
- Binary classification

## Model Used
- **Logistic Regression**

## Preprocessing Steps
- Handling missing values (none)
- Feature scaling using `StandardScaler`
- Train-Test split (80%-20%)

## Evaluation Metrics
- Accuracy: 86.7%
- Precision: 83.7%
- Confusion Matrix:  
# Phishing Email Detection using Hybrid Machine Learning and Heuristics

This project implements a hybrid phishing email detection system using a combination of **machine learning models (XGBoost, SVM)** and **heuristic rule-based techniques**. It aims to identify potentially malicious emails based on both **email content** and **URL/sender-based patterns**.

---

## Overview

Phishing attacks remain one of the top cybersecurity threats. This system aims to:
- Detect phishing attempts in email content
- Combine rule-based heuristics and machine learning for improved accuracy
- Handle both known and unknown attack vectors

---

## Features

- **Heuristic-based analysis**:
  - Detects keywords commonly used in phishing
  - Evaluates URLs and sender addresses for known red flags
- **Machine Learning models**:
  - **XGBoost** and **SVM** trained on email text features
  - Uses TF-IDF for feature extraction
  - Trained on the **CEAS 2008 dataset**

---

## Technologies & Libraries

- **Python 3**
- **scikit-learn** (SVM, TF-IDF, model evaluation)
- **xgboost** (ML model)
- **pandas**, **numpy** (data manipulation)
- **nltk** (optional text processing)
- **pickle** (model persistence)

---


