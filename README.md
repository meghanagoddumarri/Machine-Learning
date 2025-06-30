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
