import pandas as pd
from datasets import load_dataset
import os

def prepare_data():
    print("Loading dataset from Hugging Face...")
    # Using dair-ai/emotion which has joy, sadness, anger, fear, love, surprise
    dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)
    
    # Mapping IDs to labels
    # 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
    label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    for split in ['train', 'validation', 'test']:
        df = pd.DataFrame(dataset[split])
        df['label_name'] = df['label'].map(label_map)
        
        # Save raw version
        raw_path = f'data/raw/{split}.csv'
        df.to_csv(raw_path, index=False)
        print(f"Saved {split} to {raw_path}")
        
        # In this simple case, processed is same as raw (cleaning can be added if needed)
        processed_path = f'data/processed/{split}.csv'
        df.to_csv(processed_path, index=False)
        print(f"Saved {split} to {processed_path}")

if __name__ == "__main__":
    prepare_data()
