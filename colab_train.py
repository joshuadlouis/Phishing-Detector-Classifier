# =============================================================================
# PHISHING INTENT CLASSIFIER — Google Colab Training Script
# =============================================================================
#
# BEFORE running this script, do the following in separate Colab cells:
#
#   STEP A — Mount Drive (run in a cell):
#       from google.colab import drive
#       drive.mount('/content/drive')
#
#   STEP B — Upload labeled_dataset_cache.csv (run in a cell):
#       from google.colab import files
#       uploaded = files.upload()   # select labeled_dataset_cache.csv
#
#   STEP C — Install deps + run (run in a cell):
#       !pip install -q -U transformers accelerate datasets scikit-learn beautifulsoup4 tqdm
#       !python colab_train.py
#
#   STEP D — After training, download the model (run in a cell):
#       from google.colab import files
#       files.download('/content/intent_model.zip')
#
# =============================================================================

import os
import sys
import warnings
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# =============================================================================
# CONFIG — edit these paths if needed
# =============================================================================
# The script looks for the CSV in /content/ first (uploaded via sidebar),
# then falls back to Google Drive if it's mounted.
CONTENT_DIR    = Path("/content")
DRIVE_DIR      = Path("/content/drive/MyDrive/phishing_classifier")
CACHE_FILENAME = "labeled_dataset_cache.csv"
MODEL_SAVE_DIR = CONTENT_DIR / "intent_model"
ZIP_OUTPUT     = str(CONTENT_DIR / "intent_model.zip")

# Resolve cache path
if (CONTENT_DIR / CACHE_FILENAME).exists():
    CACHE_PATH = CONTENT_DIR / CACHE_FILENAME
    print(f"✅ Found cache at {CACHE_PATH}")
elif (DRIVE_DIR / CACHE_FILENAME).exists():
    CACHE_PATH = DRIVE_DIR / CACHE_FILENAME
    print(f"✅ Found cache on Drive at {CACHE_PATH}")
else:
    print("❌ ERROR: labeled_dataset_cache.csv not found.")
    print("   Upload it first by running in a Colab cell:")
    print("     from google.colab import files")
    print("     files.upload()")
    sys.exit(1)

# =============================================================================
# GPU CHECK
# =============================================================================
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {gpu_name}")
    DEVICE = 0
else:
    print("⚠️  No GPU detected — go to Runtime > Change runtime type > T4 GPU")
    DEVICE = -1

# =============================================================================
# INTENT LABELS  (7 classes — 'Liking' removed; it was a zero-shot catch-all)
# =============================================================================
INTENT_LABELS = [
    "Urgency and Scarcity",
    "Authority",
    "Fear",
    "Greed and Reciprocity",
    "Commitment and Consistency",
    "Consensus and Social Proof",
    "Safe / Neutral",
]
STALE_LABEL = "Liking"

# =============================================================================
# STEP 1 — Load cache
# =============================================================================
print(f"\n{'='*60}")
print("STEP 1 — Loading labeled dataset cache")
print(f"{'='*60}")

df = pd.read_csv(CACHE_PATH)
print(f"Loaded {len(df):,} rows  |  Columns: {df.columns.tolist()}")
print("\nCurrent intent distribution:")
print(df["intent"].value_counts())

# =============================================================================
# STEP 2 — Patch stale 'Liking' rows via Zero-Shot re-classification
# =============================================================================
print(f"\n{'='*60}")
print("STEP 2 — Patching stale 'Liking' labels")
print(f"{'='*60}")

stale_mask  = df["intent"] == STALE_LABEL
stale_count = stale_mask.sum()

if stale_count == 0:
    print(f"✅ No '{STALE_LABEL}' rows found — cache is clean, skipping re-labeling.")
else:
    print(f"Found {stale_count:,} stale '{STALE_LABEL}' rows. Re-labeling now...")

    zs_classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-3",
        device=DEVICE,
    )

    # Truncate to 256 chars — email intent is in the opening lines
    stale_texts = [str(t)[:256] if str(t).strip() else "Neutral"
                   for t in df.loc[stale_mask, "text"].tolist()]
    batch_size  = 64   # was 32 — T4 handles 64 comfortably at 256-char truncation
    new_labels  = []

    patch_ckpt = CONTENT_DIR / "patch_checkpoint.csv"
    resume_idx = 0
    if patch_ckpt.exists():
        saved_ckpt = pd.read_csv(patch_ckpt)
        new_labels = saved_ckpt["intent"].tolist()
        resume_idx = len(new_labels)
        print(f"Resuming from checkpoint at index {resume_idx}...")

    for i in tqdm(range(resume_idx, len(stale_texts), batch_size), desc="Patching labels"):
        batch   = stale_texts[i : i + batch_size]
        results = zs_classifier(batch, candidate_labels=INTENT_LABELS, batch_size=batch_size)
        if isinstance(results, dict):
            results = [results]
        new_labels.extend(r["labels"][0] for r in results)

        if len(new_labels) % 500 == 0:
            pd.DataFrame({"intent": new_labels}).to_csv(patch_ckpt, index=False)

    df.loc[stale_mask, "intent"] = new_labels
    if patch_ckpt.exists():
        patch_ckpt.unlink()

    df.to_csv(CACHE_PATH, index=False)
    print(f"\n✅ Cache updated. New distribution:")
    print(df["intent"].value_counts())

# Remove any remaining stale rows
df = df[df["intent"] != STALE_LABEL].copy()

# =============================================================================
# STEP 3 — Tokenize
# =============================================================================
print(f"\n{'='*60}")
print("STEP 3 — Tokenizing")
print(f"{'='*60}")

label_to_id = {label: i for i, label in enumerate(INTENT_LABELS)}
id_to_label = {i: label for label, i in label_to_id.items()}

df["intent_id"] = df["intent"].map(label_to_id)
before = len(df)
df.dropna(subset=["intent_id"], inplace=True)
df["intent_id"] = df["intent_id"].astype(int)
print(f"Dropped {before - len(df)} unmappable rows. Training on {len(df):,} samples.")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["intent_id"].tolist(),
    test_size=0.15,
    random_state=42,
    stratify=df["intent_id"].tolist(),
)
print(f"Train: {len(train_texts):,}  |  Val: {len(val_texts):,}")

MODEL_NAME = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizing...")
train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_enc   = tokenizer(val_texts,   truncation=True, padding=True, max_length=256)


class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item            = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"]  = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = PhishingDataset(train_enc, train_labels)
val_dataset   = PhishingDataset(val_enc,   val_labels)

# Compute class weights — inversely proportional to class frequency.
# 'balanced' = n_samples / (n_classes * class_count_per_label)
# Rare classes like 'Greed and Reciprocity' (45 samples) get high weights;
# dominant classes like 'Safe / Neutral' get low weights.
raw_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(list(range(len(INTENT_LABELS)))),
    y=train_labels,
)
class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float)

print("\nClass weights (higher = minority class given more loss penalty):")
for label, weight in zip(INTENT_LABELS, raw_weights):
    print(f"  {label:<30} {weight:.4f}")

# =============================================================================
# STEP 4 — Load model
# =============================================================================
print(f"\n{'='*60}")
print("STEP 4 — Loading DistilBERT")
print(f"{'='*60}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(INTENT_LABELS),
    id2label=id_to_label,
    label2id=label_to_id,
)
print(f"✅ Model ready — {len(INTENT_LABELS)} output classes.")

# =============================================================================
# STEP 5 — Metrics
# =============================================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds  = np.argmax(predictions, axis=1)
    report = classification_report(
        labels, preds,
        labels=list(range(len(INTENT_LABELS))),
        target_names=INTENT_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy":    report["accuracy"],
        "macro_f1":    report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


# =============================================================================
# WeightedTrainer — overrides compute_loss to use class-weighted cross-entropy
# =============================================================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        # Move weights to the same device as the model output
        weights = self.class_weights.to(logits.device)
        loss    = nn.CrossEntropyLoss(weight=weights)(logits, labels)

        return (loss, outputs) if return_outputs else loss

# =============================================================================
# STEP 6 — Train
# =============================================================================
print(f"\n{'='*60}")
print("STEP 6 — Fine-tuning")
print(f"{'='*60}")

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir                  = "/content/results",
    num_train_epochs            = 4,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 64,
    warmup_steps                = 200,
    weight_decay                = 0.01,
    learning_rate               = 2e-5,
    logging_dir                 = "/content/logs",
    logging_steps               = 50,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "macro_f1",
    fp16                        = torch.cuda.is_available(),
    report_to                   = "none",
)

trainer = WeightedTrainer(
    class_weights   = class_weights_tensor,
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    compute_metrics = compute_metrics,
)

trainer.train()

# =============================================================================
# STEP 7 — Evaluate & Save
# =============================================================================
print(f"\n{'='*60}")
print("STEP 7 — Evaluation & Saving")
print(f"{'='*60}")

eval_results = trainer.evaluate()
print("\n📊 Evaluation Results:")
for k, v in eval_results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

trainer.save_model(str(MODEL_SAVE_DIR))
tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
print(f"\n✅ Model saved to {MODEL_SAVE_DIR}")

# Copy to Drive if mounted
if DRIVE_DIR.exists():
    drive_model = DRIVE_DIR / "intent_model"
    if drive_model.exists():
        shutil.rmtree(drive_model)
    shutil.copytree(str(MODEL_SAVE_DIR), str(drive_model))
    print(f"✅ Model also saved to Drive: {drive_model}")

# Zip for download
shutil.make_archive(str(CONTENT_DIR / "intent_model"), "zip", str(MODEL_SAVE_DIR))
print(f"✅ Zipped: {ZIP_OUTPUT}")
print("\nTo download, run in a Colab cell:")
print("  from google.colab import files")
print("  files.download('/content/intent_model.zip')")

# =============================================================================
# STEP 8 — Sanity check
# =============================================================================
print(f"\n{'='*60}")
print("STEP 8 — Sanity Check")
print(f"{'='*60}")

inf_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

TEST_EMAILS = [
    "Your account will be suspended in 24 hours if you do not verify your login details.",
    "Click here to claim your $5,000 Amazon gift card winner prize!",
    "This is the IRS. You owe back taxes. A warrant will be issued if you don't pay now.",
    "As CEO, I need you to securely wire funds to our new vendor today. Keep this confidential.",
    "Hi team, here are the meeting notes from last Thursday's all-hands.",
]

for text in TEST_EMAILS:
    result = inf_pipeline(text, truncation=True, max_length=256)
    print(f"\n  Text   : {text[:75]}...")
    print(f"  Intent : {result[0]['label']}  ({result[0]['score']:.1%})")

print("\n🎉 Done!")
