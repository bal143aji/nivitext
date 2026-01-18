import joblib
import sys
import os

def run_demo():
    model_path = 'models/baseline_model.pkl'
    tfidf_path = 'models/tfidf_vectorizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        print("Model files not found. Please run scripts/train_baseline.py first.")
        return

    print("Loading models...")
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

    print("\n--- Text Emotion Classifier Demo ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Enter text to classify: ")
        if text.lower() in ['exit', 'quit']:
            break
        
        if not text.strip():
            continue

        # Transform and predict
        text_vec = tfidf.transform([text])
        prediction = model.predict(text_vec)[0]
        probs = model.predict_proba(text_vec)[0]
        
        emotion = label_map[prediction]
        confidence = probs[prediction]

        print(f"Predicted Emotion: {emotion.upper()} (Confidence: {confidence:.2%})")
        
        # Show all probabilities
        print("Probabilities:")
        for i, prob in enumerate(probs):
            print(f"  {label_map[i]}: {prob:.2%}")
        print("-" * 30)

if __name__ == "__main__":
    run_demo()
