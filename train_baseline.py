import pandas as pd
import joblib
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_baseline():
    print("Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']

    label_names = sorted(train_df['label_name'].unique())
    # Create a mapping from ID to Name for reporting
    id_to_name = dict(zip(train_df['label'], train_df['label_name']))
    target_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training Logistic Regression model...")
    # class_weight='balanced' helps with class imbalance
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/baseline_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("Model saved to models/baseline_model.pkl")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_metrics.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Baseline Confusion Matrix')
    plt.savefig('results/baseline_confusion_matrix.png')
    print("Confusion matrix saved to results/baseline_confusion_matrix.png")

if __name__ == "__main__":
    train_baseline()
