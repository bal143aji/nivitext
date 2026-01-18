import pandas as pd
import os
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1_macro}

def train_transformer():
    model_name = "distilbert-base-uncased"
    print(f"Loading {model_name} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/validation.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Prepare label labels names
    num_labels = len(train_df['label'].unique())
    id2label = {int(i): name for i, name in zip(train_df['label'], train_df['label_name'])}
    label2id = {name: int(i) for i, name in zip(train_df['label'], train_df['label_name'])}

    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Use a smaller number of epochs and batch size for CPU
    training_args = TrainingArguments(
        output_dir="./models/transformer_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1, # Setting to 1 for demonstration, usually 3-5
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=100,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print("Starting training (this may take a while on CPU)...")
    trainer.train()

    print("Evaluating on test set...")
    metrics = trainer.evaluate(tokenized_test)
    print(metrics)

    # Save final model
    os.makedirs('models/best_transformer', exist_ok=True)
    model.save_pretrained('models/best_transformer')
    tokenizer.save_pretrained('models/best_transformer')
    print("Model saved to models/best_transformer")

    with open('results/transformer_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_transformer()
