import os
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Intent Labels (Combining base intents with Cialdini's 6 Principles)
INTENT_LABELS = [
    "Urgency and Scarcity", 
    "Authority", 
    "Fear", 
    "Greed and Reciprocity", 
    "Commitment and Consistency", 
    "Consensus and Social Proof", 
    "Liking"
]

def clean_text(text):
    """Strips HTML tags and standardizes text."""
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return text.strip()

def ingest_and_preprocess_data(data_path):
    """
    Loads .csv files and iterates through folders containing .txt files.
    Appends them to a single DataFrame, handles encoding errors, and preprocesses.
    """
    data_dir = Path(data_path)
    data_frames = []
    
    print(f"Scanning directory: {data_dir.resolve()} for files...")
    
    # 1. Load CSV files
    for csv_file in data_dir.rglob("*.csv"):
        try:
            # Handle encoding errors by trying utf-8, then fallback to latin1
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip')
            
            # Map standard columns conceptually (looking for common naming)
            # Find the column that most likely contains text
            text_col = next((col for col in df.columns if col.lower() in ['text', 'body', 'email_content', 'email text', 'content']), None)
            label_col = next((col for col in df.columns if col.lower() in ['label', 'class', 'is_phishing', 'target']), None)
            
            if text_col:
                temp_df = pd.DataFrame()
                temp_df['text'] = df[text_col]
                if label_col:
                    temp_df['label'] = df[label_col].astype(str)
                else: # Fallback based on filename if possible
                    is_phishing = any(x in csv_file.name.lower() for x in ['phish', 'fraud', 'spam'])
                    temp_df['label'] = '1' if is_phishing else '0'
                data_frames.append(temp_df)
        except Exception as e:
            print(f"Warning: Could not process {csv_file.name}: {e}")
            
    # 2. Iterate through folders containing .txt files (e.g., 'Phishing' and 'Safe')
    for txt_file in data_dir.rglob("*.txt"):
        # Assign label based on the parent folder name
        parent_name = txt_file.parent.name.lower()
        if parent_name in ['phishing', 'spam', 'fraud']:
            label = '1'
        elif parent_name in ['safe', 'ham', 'legit', 'normal']:
            label = '0'
        else:
            # If in the root dataset folder with a name indicating phishing
            if any(x in txt_file.name.lower() for x in ['phish', 'fraud', 'spam']):
                label = '1'
            else:
                continue # Skip unidentifiable labels
                
        try:
            # Handle encoding errors during file reading
            with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            data_frames.append(pd.DataFrame({'text': [content], 'label': [label]}))
        except Exception as e:
            print(f"Warning: Could not read {txt_file.name}: {e}")
            
    if not data_frames:
        raise ValueError("No data could be loaded. Please check the dataset directory.")
        
    main_df = pd.concat(data_frames, ignore_index=True)
    
    # 3. Preprocessing
    print("Preprocessing: Cleaning HTML and removing duplicates...")
    main_df['text'] = main_df['text'].apply(clean_text)
    
    # Keep only reasonably long text and explicitly phishing items
    main_df = main_df[main_df['text'].str.len() > 20]
    
    # Remove duplicate entries
    before_drop = len(main_df)
    main_df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Dropped {before_drop - len(main_df)} duplicate rows. Total unique items: {len(main_df)}")
    
    return main_df

def label_phishing_intents(df, sample_limit=None):
    """
    Uses Zero-Shot classification to assign Intent labels to Phishing text.
    """
    print("Loading Zero-Shot Classifier (facebook/bart-large-mnli)...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    
    # Filter for phishing emails (assuming '1' or 'phishing' denotes phishing)
    phishing_df = df[df['label'].astype(str).isin(['1', 'phishing', 'spam', 'fraud'])].copy()
    
    if len(phishing_df) == 0:
        raise ValueError("No phishing examples found to label intents. Check label mapping.")
        
    if sample_limit:
        print(f"Downsampling to {sample_limit} examples for demonstration.")
        phishing_df = phishing_df.sample(min(sample_limit, len(phishing_df)), random_state=42)
        
    from tqdm import tqdm
    intent_labels = []
    
    print(f"Assigning 'Intent' labels for {len(phishing_df)} items...")
    for idx, row in tqdm(phishing_df.iterrows(), total=len(phishing_df), desc="Zero-Shot Pipeline"):
        # Truncate text context slightly to prevent massive memory usage
        text_context = row['text'][:800] 
        result = classifier(text_context, candidate_labels=INTENT_LABELS)
        best_intent = result['labels'][0]
        intent_labels.append(best_intent)
        
    phishing_df['intent'] = intent_labels
    return phishing_df

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Generate sklearn classification report as a dictionary
    # Provide explicitly the labels list to avoid crashes if some batches don't contain all classes
    report = classification_report(
        labels, 
        preds, 
        labels=list(range(len(INTENT_LABELS))), 
        target_names=INTENT_LABELS, 
        output_dict=True,
        zero_division=0
    )
    
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

def train_intent_classifier(phishing_df, model_save_path="./intent_model"):
    """
    Fine-tunes distilbert-base-uncased for intent classification.
    """
    print("Preparing dataset for fine-tuning...")
    
    # Create mapping
    label_to_id = {label: i for i, label in enumerate(INTENT_LABELS)}
    phishing_df['intent_id'] = phishing_df['intent'].map(label_to_id)
    
    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        phishing_df['text'].tolist(), 
        phishing_df['intent_id'].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(INTENT_LABELS),
        id2label={i: l for l, i in label_to_id.items()},
        label2id=label_to_id
    )
    
    # Create HuggingFace Datasets
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    
    class PhishingDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
            
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
            
        def __len__(self):
            return len(self.labels)
            
    train_dataset = PhishingDataset(train_encodings, train_labels)
    val_dataset = PhishingDataset(val_encodings, val_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Fine-tuning DistilBERT...")
    trainer.train()
    
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    print(f"Saving fine-tuned model to {model_save_path}...")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("Model saved successfully.")

def validate_and_infer(model_path, custom_strings):
    """
    Loads saved model and runs inference on custom strings.
    """
    print(f"\n--- Validation & Inference ---")
    print(f"Loading custom model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Use HF Pipeline for easy inference
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    for text in custom_strings:
        result = classifier(text, truncation=True, max_length=512)
        print(f"\nText: {text}")
        print(f"Predicted Intent: {result[0]['label']} (Confidence: {result[0]['score']:.4f})")

if __name__ == "__main__":
    # --- Configuration ---
    # In Windows, we use pathlib for safety
    DATASET_PATH = Path("rokibulroni_phishing_dataset")
    SAVE_MODEL_DIR = "./intent_model"
    
    # 1. Ingestion and Preprocessing
    df_raw = ingest_and_preprocess_data(DATASET_PATH)
    
    # 2. Assigning Intents using Zero-Shot (Limiting to 100 for demonstration speed)
    # Note: Remove sample_limit to process the entire dataset (may take long without GPU).
    df_intents = label_phishing_intents(df_raw, sample_limit=100)
    
    # 3. Model Architecture & Fine-tuning
    train_intent_classifier(df_intents, model_save_path=SAVE_MODEL_DIR)
    
    # 4. Validation
    custom_tests = [
        "Your account will be suspended in 24 hours if you do not verify your login details immediately.",
        "Click here to claim your $5,000 Amazon gift card winner prize!",
        "This is the IRS. You owe back taxes and a warrant will be issued for your arrest if you don't pay now.",
        "As the CEO of the company, I need you to securely wire funds to our new vendor today."
    ]
    validate_and_infer(SAVE_MODEL_DIR, custom_tests)
