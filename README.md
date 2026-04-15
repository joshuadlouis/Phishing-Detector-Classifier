# Phishing Intent Classifier

## Overview

The **Phishing Intent Classifier** is an NLP pipeline designed to categorize malicious emails and texts based on the psychological manipulation tactic they employ.

Rather than a binary spam filter ("Safe or Phishing?"), this system identifies the underlying behavioral vector of an attack by mapping each communication to one of **Cialdini's 6 Principles of Persuasion**. The goal is to produce actionable threat intelligence — distinguishing, for example, a targeted authority-based spear-phishing attempt from broad urgency-driven spam.

The classifier assigns one of the following 7 intent categories:

| Label | Description |
|---|---|
| `Urgency and Scarcity` | Creates time pressure or limited-availability pressure to force immediate action |
| `Authority` | Impersonates figures of trust — executives, institutions, government agencies |
| `Fear` | Threatens negative consequences such as account suspension, legal action, or arrest |
| `Greed and Reciprocity` | Promises financial reward, prizes, or exploits obligation ("you owe us") |
| `Commitment and Consistency` | Leverages prior agreement or identity to compel continued compliance |
| `Consensus and Social Proof` | Manufactures legitimacy through implied group approval or urgency |
| `Safe / Neutral` | Benign communication with no detectable manipulation vector |

> **Note on label selection:** An eighth candidate label, `Liking`, was evaluated during initial zero-shot labeling but removed from the final label set. The zero-shot model assigned it to ~84% of phishing samples, rendering it a non-discriminative catch-all. All `Liking`-assigned rows were subsequently re-labeled using the 7-class schema.

---

## Repository Structure

```
.
├── build_classifier.py       # Local training pipeline (ingestion → labeling → fine-tuning)
├── colab_train.py            # Google Colab training script (GPU-optimized, resumable)
├── app.py                    # Flask inference server
├── benchmark_model.py        # Latency and accuracy benchmarking
├── labeled_dataset_cache.csv # Zero-shot labeled dataset (auto-generated, ~170MB)
└── intent_model/             # Trained DistilBERT artifact
    ├── config.json           # Architecture definition and label mappings
    ├── model.safetensors     # Fine-tuned model weights (~267MB)
    ├── tokenizer.json        # Full tokenizer in fast (Rust-based) format
    ├── tokenizer_config.json # Tokenizer settings
    └── training_args.bin     # Training configuration snapshot
```

---

## Training Pipeline

The system uses a three-stage transfer-learning workflow.

### 1. Data Ingestion

Raw data is sourced from the `rokibulroni/Phishing-Email-Dataset`. The ingestion script (`build_classifier.py`) recursively scans the dataset directory using `pathlib`, loading both structured `.csv` files and raw `.txt` files. HTML tags are stripped via `BeautifulSoup`, duplicate entries are dropped, and label columns are normalized to a binary `0` (safe) / `1` (phishing) schema.

### 2. Zero-Shot Intent Labeling

Because the source dataset only provides binary labels, intent assignment is performed via zero-shot classification. The `valhalla/distilbart-mnli-12-3` model is used as the labeler — a distilled BART variant that runs the 7 candidate labels against each text using Natural Language Inference (NLI). Safe emails bypass the zero-shot step entirely and are tagged `Safe / Neutral` directly.

Results are cached to `labeled_dataset_cache.csv`. If the cache already exists, the pipeline performs a targeted patch — re-labeling only rows flagged as stale — rather than re-processing the full dataset.

### 3. Fine-Tuning

The zero-shot labels serve as ground truth for fine-tuning `distilbert-base-uncased` via Hugging Face's `Trainer` API. Key training parameters:

| Parameter | Value |
|---|---|
| Epochs | 4 |
| Train batch size | 32 |
| Max sequence length | 256 tokens |
| Learning rate | 2e-5 |
| Best model metric | `macro_f1` (balances minority classes) |
| Mixed precision | fp16 (GPU only) |

The result is the `./intent_model/` artifact — a compiled, production-ready classifier that runs in approximately 20ms on a standard CPU.

---

## Running on Google Colab (Recommended)

Training on a T4 GPU is significantly faster than local CPU/AMD GPU execution. The `colab_train.py` script handles the full pipeline including cache patching, tokenization, training, and model export.

**Setup (run each in a separate Colab cell, in order):**

```python
# Cell 1 — Install dependencies
!pip install -q -U transformers accelerate datasets scikit-learn beautifulsoup4 tqdm
```

```python
# Cell 2 — Upload the labeled dataset cache
from google.colab import files
uploaded = files.upload()   # select labeled_dataset_cache.csv
```

```python
# Cell 3 — (Optional) Authenticate with Hugging Face for higher rate limits
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'))
```

```python
# Cell 4 — Run training
!python colab_train.py
```

```python
# Cell 5 — Download the trained model
from google.colab import files
files.download('/content/intent_model.zip')
```

> The script is resumable. If the session is interrupted, it will restore progress from `patch_checkpoint.csv` and continue from where it left off.

---

## Local Training

For local execution, run `build_classifier.py` directly. DirectML is supported for AMD GPUs on Windows:

```bash
python build_classifier.py
```

The script will auto-detect CUDA, DirectML (AMD), or fall back to CPU. The labeled cache is reused across runs.

---

## Inference Server

After training is complete, the Flask application serves the model for real-time inference:

```bash
python app.py
```

The server runs at `http://127.0.0.1:5000` and accepts:

- **Plain text** (JSON body or form field)
- **File uploads**: `.txt`, `.eml`, `.pdf`, `.docx`

**Example response:**

```json
{
  "is_phishing": true,
  "verdict": "Phishing Attempt Detected",
  "intent": "Authority",
  "confidence": "94.31"
}
```

---

## Model Artifact

The trained artifact is saved to `./intent_model/` and loaded directly by the Flask inference server. The key files are:

| File | Description |
|---|---|
| `model.safetensors` | Fine-tuned DistilBERT weights (~267MB) |
| `config.json` | Architecture config including `id2label` and `label2id` mappings for all 7 classes |
| `tokenizer.json` | Self-contained fast tokenizer (HuggingFace unified format — includes full vocabulary and special tokens) |
| `tokenizer_config.json` | Tokenizer runtime settings |
| `training_args.bin` | Snapshot of the training configuration used to produce this checkpoint |

> The tokenizer uses the modern fast format (`tokenizer.json`), which is a self-contained file covering vocabulary, special tokens, and tokenization rules. Legacy files such as `vocab.txt` and `special_tokens_map.json` are not required when this format is present and `AutoTokenizer` is used for loading.

---

## Dataset

Source: [`rokibulroni/Phishing-Email-Dataset`](https://huggingface.co/datasets/rokibulroni/Phishing-Email-Dataset)

| Split | Count |
|---|---|
| Phishing | ~42,256 |
| Safe / Neutral | ~39,510 |
| Total (after deduplication) | ~81,766 |
