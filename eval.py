import pandas as pd
import joblib
import json
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_and_analyze():
    print("Loading data and baseline model...")
    test_df = pd.read_csv('data/processed/test.csv')
    model = joblib.load('models/baseline_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')

    X_test = test_df['text']
    y_test = test_df['label']
    
    # Mapping
    id_to_name = dict(zip(test_df['label'], test_df['label_name']))
    target_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    print("Generating predictions...")
    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    y_probs = model.predict_proba(X_test_tfidf)

    # Detailed report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification Report:")
    print(report)

    # Error Analysis
    print("Performing error analysis...")
    test_df['predicted'] = y_pred
    test_df['predicted_name'] = test_df['predicted'].map(id_to_name)
    test_df['confidence'] = [y_probs[i][y_pred[i]] for i in range(len(y_pred))]
    
    errors = test_df[y_test != y_pred].copy()

    # Sample some typical errors
    error_samples = errors.sort_values(by='confidence', ascending=False).head(20)
    
    os.makedirs('results', exist_ok=True)
    error_samples.to_csv('results/error_analysis_samples.csv', index=False)
    
    with open('results/detailed_report.txt', 'w') as f:
        f.write("Baseline Model Detailed Evaluation\n")
        f.write("==================================\n\n")
        f.write(report)
        f.write("\n\nTop 20 Misclassified Samples (Sorted by Confidence):\n")
        f.write(error_samples[['text', 'label_name', 'predicted_name', 'confidence']].to_string())

    print("Results saved to results/detailed_report.txt and results/error_analysis_samples.csv")

if __name__ == "__main__":
    evaluate_and_analyze()
