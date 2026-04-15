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

# Intent Labels (Cialdini's 6 Principles minus 'Liking', which acted as a catch-all)
INTENT_LABELS = [
    "Urgency and Scarcity",
    "Authority",
    "Fear",
    "Greed and Reciprocity",
    "Commitment and Consistency",
    "Consensus and Social Proof",
    "Safe / Neutral"
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

def _get_classifier_device():
    """Returns the best available device for the zero-shot pipeline."""
    if torch.cuda.is_available():
        return 0
    try:
        import torch_directml
        if torch_directml.is_available():
            dml = torch_directml.device()
            print(f"DirectML active! Using Radeon Graphics: {torch_directml.device_name(dml.index)}")
            return dml
    except ImportError:
        pass
    return -1


def _run_zero_shot_batch(classifier, texts, resume_index=0, checkpoint_path=None):
    """Runs zero-shot classification in batches with checkpoint support."""
    from tqdm import tqdm

    batch_size = 16
    intent_labels = []
    print(f"Running zero-shot on {len(texts) - resume_index} items...")

    for i in tqdm(range(resume_index, len(texts), batch_size), desc="Zero-Shot Labeling"):
        batch_texts = texts[i : i + batch_size]
        valid_batch = [t if t.strip() else "Neutral" for t in batch_texts]

        results = classifier(valid_batch, candidate_labels=INTENT_LABELS, batch_size=batch_size)
        if isinstance(results, dict):
            results = [results]

        for res in results:
            intent_labels.append(res['labels'][0])

        # Checkpoint every ~100 items
        if checkpoint_path and len(intent_labels) % 100 == 0:
            pd.DataFrame({'intent': intent_labels}).to_csv(checkpoint_path, index=False)

    return intent_labels


def label_phishing_intents(df, sample_limit=None):
    """
    Uses Zero-Shot classification to assign Intent labels to phishing text.
    - Caches results to 'labeled_dataset_cache.csv'.
    - If the cache exists but contains stale 'Liking' rows (removed label),
      only those rows are re-classified — saving significant compute time.
    """
    cache_path = Path("labeled_dataset_cache.csv")
    checkpoint_path = Path("labeled_dataset_checkpoint.csv")
    STALE_LABEL = "Liking"

    # ------------------------------------------------------------------ #
    # SMART CACHE PATCH: re-label only stale rows, keep everything else   #
    # ------------------------------------------------------------------ #
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        stale_mask = cached_df['intent'] == STALE_LABEL
        stale_count = stale_mask.sum()

        if stale_count == 0:
            print(f"\n--- [CACHE FOUND, NO STALE ROWS] ---")
            print(f"Loaded {len(cached_df):,} rows from {cache_path}.")
            return cached_df

        print(f"\n--- [CACHE PATCH REQUIRED] ---")
        print(f"Found {stale_count:,} rows labeled '{STALE_LABEL}' (removed label).")
        print(f"Re-labeling these rows only — no need to re-process the full dataset.")

        device = _get_classifier_device()
        print("Loading Zero-Shot Classifier (valhalla/distilbart-mnli-12-3)...")
        classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=device)

        stale_df = cached_df[stale_mask].copy()
        texts = [str(t)[:800] for t in stale_df['text'].tolist()]

        # Check for a partial checkpoint scoped to this patch run
        resume_index = 0
        patch_checkpoint = Path("patch_checkpoint.csv")
        if patch_checkpoint.exists():
            ckpt = pd.read_csv(patch_checkpoint)
            resume_index = len(ckpt)
            print(f"Resuming patch from index {resume_index}...")

        new_labels = _run_zero_shot_batch(
            classifier, texts,
            resume_index=resume_index,
            checkpoint_path=patch_checkpoint
        )

        # Combine with previously checkpointed labels if resuming
        if resume_index > 0:
            prev_labels = pd.read_csv(patch_checkpoint)['intent'].tolist()[:resume_index]
            new_labels = prev_labels + new_labels

        cached_df.loc[stale_mask, 'intent'] = new_labels
        cached_df.to_csv(cache_path, index=False)
        if patch_checkpoint.exists():
            patch_checkpoint.unlink()

        print(f"\nUpdated intent distribution:")
        print(cached_df['intent'].value_counts())
        return cached_df

    # ------------------------------------------------------------------ #
    # FULL LABELING RUN (no cache exists)                                  #
    # ------------------------------------------------------------------ #
    device = _get_classifier_device()
    print("Loading Zero-Shot Classifier (valhalla/distilbart-mnli-12-3)...")
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=device)

    phishing_mask = df['label'].astype(str).isin(['1', 'phishing', 'spam', 'fraud'])
    phishing_df = df[phishing_mask].copy()
    safe_df = df[~phishing_mask].copy()
    safe_df['intent'] = "Safe / Neutral"

    if sample_limit:
        print(f"Downsampling to {sample_limit} examples for demonstration.")
        phishing_df = phishing_df.sample(min(sample_limit, len(phishing_df)), random_state=42)
        safe_df = safe_df.sample(min(sample_limit, len(safe_df)), random_state=42)

    if len(phishing_df) == 0:
        raise ValueError("No phishing examples found to label intents.")

    # RESUME LOGIC
    resume_index = 0
    if checkpoint_path.exists():
        resume_index = len(pd.read_csv(checkpoint_path))
        print(f"\n--- [RESUMING FROM CHECKPOINT] ---")
        print(f"Already labeled {resume_index} items. Picking up where we left off...")

    texts = [str(row['text'])[:800] for _, row in phishing_df.iterrows()]
    intent_labels = _run_zero_shot_batch(
        classifier, texts,
        resume_index=resume_index,
        checkpoint_path=checkpoint_path
    )

    if resume_index > 0:
        prev_labels = pd.read_csv(checkpoint_path)['intent'].tolist()[:resume_index]
        intent_labels = prev_labels + intent_labels

    phishing_df['intent'] = intent_labels
    combined_df = pd.concat([safe_df, phishing_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    combined_df.to_csv(cache_path, index=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nFinal intent distribution:")
    print(combined_df['intent'].value_counts())
    return combined_df

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
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
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
    
    # 2. Assigning Intents using Zero-Shot across the ENTIRE dataset
    # Note: This processes all ~42k phishing items using your Radeon GPU.
    df_intents = label_phishing_intents(df_raw, sample_limit=None)
    
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
