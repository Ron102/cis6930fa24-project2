import pandas as pd
import string
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


def preprocessor(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    preprocessed_text = ' '.join(words)

    name_len = [len(block) for block in text.split() if '█' in block]
    redact_features = {
        "max_redaction_length": max(name_len, default=0),
        "mean_redaction_length": np.mean(name_len) if name_len else 0
    }
    return preprocessed_text, redact_features


def load_data(unredactor_tsv_path, test_tsv_path, max_samples=5000):
    df = pd.read_csv(unredactor_tsv_path, sep='\t', on_bad_lines="skip")
    df.columns = ["Type", "Name", "Context"]
    train_data = df[df['Type'] == 'training']
    validation_data = df[df['Type'] == 'validation']

    train_data = train_data.sample(n=min(max_samples, len(train_data)), random_state=42)
    validation_data = validation_data.sample(n=min(max_samples, len(validation_data)), random_state=42)

    X_train_text = []
    X_train_features = []
    for context in train_data['Context']:
        text, features = preprocessor(context)
        X_train_text.append(text)
        X_train_features.append(features)

    X_val_text = []
    X_val_features = []
    for context in validation_data['Context']:
        text, features = preprocessor(context)
        X_val_text.append(text)
        X_val_features.append(features)

    y_train = train_data['Name'].tolist()
    y_val = validation_data['Name'].tolist()

    test_info = pd.read_csv(test_tsv_path, sep='\t', header=None)
    
    test_info.columns = ["id", "context"]

    test_texts = []
    test_features = []
    for context in test_info['context']:
        text, features = preprocessor(context)
        test_texts.append(text)
        test_features.append(features)

    return X_train_text, X_train_features, y_train, X_val_text, X_val_features, y_val, test_info, test_texts, test_features


def submission_output(test_info, predictions, output_file="submission.tsv"):
    submission = pd.DataFrame({
        'id': test_info['id'],
        'name': predictions 
    })
    submission = submission[['id', 'name']]

    submission.to_csv(output_file, sep='\t', index=False, header=False)



def train_eval_model(X_train_text, X_train_features, y_train, X_val_text, X_val_features, y_val):
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 3))  
    scaler = StandardScaler()

    transformer = ColumnTransformer(
        transformers=[
            ('tfidf', tfidf, 'text'),
            ('scaler', scaler, ['max_redaction_length', 'mean_redaction_length'])
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ('features', transformer),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight="balanced"))
        ]
    )

    train_data = pd.DataFrame({
        'text': X_train_text,
        'max_redaction_length': [f['max_redaction_length'] for f in X_train_features],
        'mean_redaction_length': [f['mean_redaction_length'] for f in X_train_features]
    })

    val_data = pd.DataFrame({
        'text': X_val_text,
        'max_redaction_length': [f['max_redaction_length'] for f in X_val_features],
        'mean_redaction_length': [f['mean_redaction_length'] for f in X_val_features]
    })

    model_pipeline.fit(train_data, y_train)

    y_pred_val = model_pipeline.predict(val_data)

    accuracy = accuracy_score(y_val, y_pred_val)
    precision = precision_score(y_val, y_pred_val, average="weighted", zero_division=0)
    recall = recall_score(y_val, y_pred_val, average="weighted", zero_division=0)
    f1 = f1_score(y_val, y_pred_val, average="weighted", zero_division=0)

    print(f"Validation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    return model_pipeline


def main():
    unredactor_tsv_path, test_tsv_path = Path("unredactor.tsv"), Path("test.tsv") 

    X_train_text, X_train_features, y_train, X_val_text, X_val_features, y_val, test_info, X_test_text, X_test_features = load_data(
        unredactor_tsv_path, test_tsv_path, max_samples=5000
    )

    model = train_eval_model(X_train_text, X_train_features, y_train, X_val_text, X_val_features, y_val)


    test_data = pd.DataFrame({
        'text': X_test_text,
        'max_redaction_length': [f['max_redaction_length'] for f in X_test_features],
        'mean_redaction_length': [f['mean_redaction_length'] for f in X_test_features]
    })

    predictions = model.predict(test_data)

    submission_output(test_info, predictions)


if __name__ == "__main__":
    main()
