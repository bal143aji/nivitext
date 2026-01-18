# Project Report: Text Emotion Classification

## 1. Dataset Summary
- **Source**: `dair-ai/emotion` (available via Hugging Face).
- **Size**: 16,000 training, 2,000 validation, 2,000 test examples.
- **Labels**: Sadness, Joy, Love, Anger, Fear, Surprise.
- **Characteristics**: Short English sentences from social media platforms.

## 2. Models and Hyperparameters

### Baseline: TF-IDF + Logistic Regression
- **Features**: TF-IDF (1-2 ngrams), max 50,000 features.
- **Classifier**: Logistic Regression (`C=1.0`, `class_weight='balanced'`).
- **Rationale**: Fast, interpretable, and effective for high-dimensional text data.

### Advanced: DistilBERT
- **Architecture**: `distilbert-base-uncased`.
- **Learning Rate**: 2e-5, Batch Size: 8, Epochs: 1 (for demo purposes).
- **Rationale**: Leverages pretrained semantic knowledge to capture context better than bag-of-words models.

## 3. Results and Analysis

### Performance Metrics (Baseline)
- **Accuracy**: 86.25%
- **Macro F1**: 0.83
- **Per-Class Highlights**:
    - **Sadness** (F1: 0.89) and **Joy** (F1: 0.88) were most accurately identified.
    - **Surprise** (F1: 0.70) remains the hardest class due to limited support (66 samples) and overlap with Joy/Fear.

### Error Analysis
Based on the top 20 misclassified samples:
1. **Ambiguity**: Sentences like "i am feeling a bit weird" could be surprise, fear, or sadness depending on context.
2. **Sarcasm**: Sarcastic expressions of "joy" are often classified literally by the baseline model.
3. **Compound Emotions**: "i feel overwhelmed" often triggers fear, but could be related to sadness or even surprise.

## 4. Ethical Considerations and Limitations
- **Data Bias**: The dataset is sourced from social media, which may favor certain dialects or expressions of emotion over others.
- **Dual Use**: While useful for customer feedback, such models can be misused for emotional manipulation in advertising.
- **Mitigation**: We used `class_weight='balanced'` to ensure minority emotions aren't ignored, but manual review is recommended for low-confidence predictions.

## 5. Future Work
- **Multi-label Classification**: Allowing a sentence to be both "fear" and "anger".
- **Explainability**: Using SHAP or LIME to visualize which tokens (e.g., "awful", "wonderful") drive the predictions.
- **Data Augmentation**: Using back-translation to increase the number of "surprise" and "love" samples.
