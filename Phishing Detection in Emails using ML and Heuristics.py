Python 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import tldextract
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class PhishingDetector:
    def _init_(self, decision_threshold=0.65):
        self.heuristic_rules = self.load_heuristic_rules()
        self.xgb_model = None
        self.svm_model = None
        self.vectorizer = None
        self.decision_threshold = decision_threshold
        self.feature_names = [
            'num_links', 'has_links', 'text_length',
            'num_suspicious_keywords', 'suspicious_sender',
            'subject_urgency'
        ]
        self.initialize_models()

    def initialize_models(self):
        try:
            with open('phishing_xgb_model.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
            with open('phishing_svm_model.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)
            with open('phishing_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Models loaded successfully")
        except:
            print("Models not found, training new models...")
            self.train_models()

    def train_models(self):
        # Load CEAS_08 dataset with robust parsing
        try:
            df = pd.read_csv(r"/content/CEAS_08.csv",
                            quoting=csv.QUOTE_ALL,
                            on_bad_lines='skip',
                            engine='python')
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return

        # Assuming columns: 'sender', 'subject', 'body', 'urls', 'label'
        df = df.rename(columns={'body': 'email_text', 'label': 'is_phishing'})
        df['has_attachments'] = 0  # CEAS_08 doesn't typically include attachments, assume none
        df = self.extract_features(df)

        # Prepare features
        X_text = df['email_text'].fillna('')
        X_other = df[self.feature_names]
        y = df['is_phishing']

        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        text_features = self.vectorizer.fit_transform(X_text).toarray()

        # Combine features
        X_combined = np.hstack([text_features, X_other.values])

        # Train XGBoost model
        self.xgb_model = XGBClassifier()
        self.xgb_model.fit(X_combined, y)

        # Train SVM model
        self.svm_model = SVC(probability=True, kernel='rbf')
        self.svm_model.fit(X_combined, y)

        # Save models
        with open('phishing_xgb_model.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)
        with open('phishing_svm_model.pkl', 'wb') as f:
            pickle.dump(self.svm_model, f)
        with open('phishing_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print("Models trained and saved successfully")

    def extract_features(self, df):
        # URL features
        df['num_links'] = df['email_text'].apply(
            lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(x))))
        df['has_links'] = (df['num_links'] > 0).astype(int)

        # Text features
        df['text_length'] = df['email_text'].apply(lambda x: len(str(x)))
        df['num_suspicious_keywords'] = df['email_text'].apply(
            lambda x: sum(1 for word in self.heuristic_rules['suspicious_keywords'] if word.lower() in str(x).lower()))

        # Sender features
        df['sender_domain'] = df['sender'].apply(lambda x: x.split('@')[-1] if '@' in str(x) else '')
        df['suspicious_sender'] = df['sender_domain'].apply(
            lambda x: 1 if x in self.heuristic_rules['suspicious_domains'] else 0)

        # Subject features
        df['subject_urgency'] = df['subject'].apply(
            lambda x: 1 if any(word in str(x).lower() for word in ['urgent', 'immediate', 'action required', 'verify']) else 0)

        return df

    def heuristic_profiler(self, email_data):
        risk_score = 0
        flags = []

        email_text = str(email_data['email_text']).lower()
        subject_text = str(email_data['subject']).lower()

        for keyword in self.heuristic_rules['suspicious_keywords']:
            if keyword in email_text or keyword in subject_text:
                risk_score += 5
                flags.append(f"Suspicious keyword: {keyword}")

        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(email_data['email_text']))
        if urls:
            risk_score += 10
            flags.append("Contains links")

        sender_domain = email_data['sender'].split('@')[-1] if '@' in email_data['sender'] else ''
        if sender_domain in self.heuristic_rules['suspicious_domains']:
            risk_score += 30
            flags.append(f"Suspicious sender: {sender_domain}")

        if email_data.get('has_attachments', False):
            risk_score += 5
            flags.append("Has attachments")

        risk_score = min(100, risk_score)

        return {
            'risk_score': risk_score,
            'flags': flags,
            'heuristic_decision': 1 if risk_score >= 50 else 0
        }

    def ml_classifier(self, email_data, model_type='xgb'):
        df = pd.DataFrame([email_data])
        df = self.extract_features(df)

        text_features = self.vectorizer.transform(df['email_text'].fillna('')).toarray()
        other_features = df[self.feature_names].values
        features = np.hstack([text_features, other_features])

        model = self.xgb_model if model_type == 'xgb' else self.svm_model
        if hasattr(model, 'n_features_in_'):
            if features.shape[1] != model.n_features_in_:
                if features.shape[1] < model.n_features_in_:
                    padding = np.zeros((features.shape[0], model.n_features_in_ - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :model.n_features_in_]

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return {
            'ml_prediction': prediction,
            'ml_confidence': probability,
            'ml_features': df[self.feature_names].to_dict('records')[0]
        }

    def load_heuristic_rules(self):
        return {
            "suspicious_keywords": ["urgent", "verify", "account", "suspended", "compromised"],
            "suspicious_domains": ["secure-login.net", "account-verify.com"],
            "shortened_url_services": ["bit.ly", "goo.gl", "tinyurl.com"],
            "ip_address_urls": True,
            "https_missing": True,
            "domain_age_threshold": 30,
            "typosquatting_check": True,
            "check_domain_age": True
        }

    def analyze_email(self, email_data):
        heuristic = self.heuristic_profiler(email_data)
        xgb_ml = self.ml_classifier(email_data, model_type='xgb')

        if heuristic['heuristic_decision'] == 1 and xgb_ml['ml_prediction'] == 1:
            decision = 1
            reason = "Both heuristic and XGBoost agree: phishing"
        elif xgb_ml['ml_confidence'] >= self.decision_threshold:
            decision = xgb_ml['ml_prediction']
            reason = f"High XGBoost confidence ({xgb_ml['ml_confidence']:.2f})"
        elif heuristic['risk_score'] >= 70:
            decision = 1
            reason = f"High heuristic risk ({heuristic['risk_score']})"
        else:
            combined = (heuristic['risk_score']/100 * 0.4) + (xgb_ml['ml_confidence'] * 0.6)
            decision = 1 if combined >= 0.5 else 0
            reason = f"Combined score: {combined:.2f}"

        return {
            'decision': decision,
            'reason': reason,
            'heuristic': heuristic,
            'xgb_ml': xgb_ml
        }

    def evaluate_models(self, test_df):
        results = {
            'true': [],
            'heuristic': {'preds': [], 'probs': []},
            'xgb': {'preds': [], 'probs': []},
            'svm': {'preds': [], 'probs': []},
            'hybrid': {'preds': [], 'probs': []}
        }

        for _, email in test_df.iterrows():
            email_data = email.to_dict()
            if 'is_phishing' not in email_data:
                continue

            results['true'].append(email_data['is_phishing'])

            h_result = self.heuristic_profiler(email_data)
            results['heuristic']['preds'].append(h_result['heuristic_decision'])
            results['heuristic']['probs'].append(h_result['risk_score']/100)

            xgb_result = self.ml_classifier(email_data, model_type='xgb')
            results['xgb']['preds'].append(xgb_result['ml_prediction'])
            results['xgb']['probs'].append(xgb_result['ml_confidence'])

            svm_result = self.ml_classifier(email_data, model_type='svm')
            results['svm']['preds'].append(svm_result['ml_prediction'])
            results['svm']['probs'].append(svm_result['ml_confidence'])

            hybrid_result = self.analyze_email(email_data)
            results['hybrid']['preds'].append(hybrid_result['decision'])
            results['hybrid']['probs'].append(
                (h_result['risk_score']/100 * 0.4) + (xgb_result['ml_confidence'] * 0.6)
            )

        metrics = {}
        y_true = np.array(results['true'])

        for approach in ['heuristic', 'xgb', 'svm', 'hybrid']:
            y_pred = np.array(results[approach]['preds'])
            metrics[approach.title()] = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1 Score': f1_score(y_true, y_pred, zero_division=0),
                'ROC AUC': roc_auc_score(y_true, results[approach]['probs'])
                           if len(np.unique(y_true)) > 1 else np.nan
            }

        return pd.DataFrame(metrics).T

# Load and prepare data
import csv  # Import csv for quoting
try:
    df = pd.read_csv(r"/content/CEAS_08.csv",
                     quoting=csv.QUOTE_ALL,
                     on_bad_lines='skip',
                     engine='python')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

df = df.rename(columns={'body': 'email_text', 'label': 'is_phishing'})
df['has_attachments'] = 0

# Split into train and test (for evaluation)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Initialize and train detector
detector = PhishingDetector()

# Evaluate on test set
print("\n=== Evaluating on Test Set ===")
metrics_df = detector.evaluate_models(test_df)
print("\n=== Performance Metrics Comparison ===")
print(metrics_df)

# Calculate hybrid advantage
base_performance = metrics_df.loc[['Xgb', 'Svm']].min()
hybrid_advantage = metrics_df.loc['Hybrid'] - base_performance
print("\n=== Hybrid Approach Advantages ===")
print(hybrid_advantage[hybrid_advantage > 0].dropna().to_string())