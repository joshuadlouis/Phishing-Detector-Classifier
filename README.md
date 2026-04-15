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
├── RESULTS.md                # Full benchmark and training results with analysis
├── templates/
│   └── index.html            # Web UI for the inference server
└── intent_model/             # Trained DistilBERT artifact (will be hosted on HuggingFace Hub — see below)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_args.bin
```

> `labeled_dataset_cache.csv` (~170MB) and `intent_model/` (~267MB) are not tracked in this repository due to size constraints. The model will hosted on HuggingFace Hub (see Model Artifact section).

---

## Training Pipeline

The system uses a three-stage transfer-learning workflow.

### 1. Data Ingestion

Raw data is sourced from the `rokibulroni/Phishing-Email-Dataset`. The ingestion script (`build_classifier.py`) recursively scans the dataset directory using `pathlib`, loading both structured `.csv` files and raw `.txt` files. HTML tags are stripped via `BeautifulSoup`, duplicate entries are dropped, and label columns are normalized to a binary `0` (safe) / `1` (phishing) schema.

### 2. Zero-Shot Intent Labeling

Because the source dataset only provides binary labels, intent assignment is performed via zero-shot classification. The `valhalla/distilbart-mnli-12-3` model is used as the labeler — a distilled BART variant that runs the 7 candidate labels against each text using Natural Language Inference (NLI). Safe emails bypass the zero-shot step entirely and are tagged `Safe / Neutral` directly.

Results are cached to `labeled_dataset_cache.csv`. If the cache already exists, the pipeline performs a targeted patch — re-labeling only rows flagged as stale — rather than re-processing the full dataset.

### 3. Fine-Tuning

The zero-shot labels serve as ground truth for fine-tuning `distilbert-base-uncased` via Hugging Face's `Trainer` API. A custom `WeightedTrainer` subclass overrides `compute_loss` to apply class-weighted cross-entropy loss, assigning higher penalty to minority classes (e.g. `Greed and Reciprocity` with ~45 training samples) to improve `macro_f1` across all classes.

| Parameter | Value |
|---|---|
| Epochs | 4 |
| Train batch size | 32 |
| Max sequence length | 256 tokens |
| Learning rate | 2e-5 |
| Loss function | `CrossEntropyLoss(weight=class_weights)` |
| Class weights | `sklearn` balanced (inverse frequency) |
| Best model metric | `macro_f1` |
| Mixed precision | fp16 (GPU only) |

The result is the `./intent_model/` artifact — a production-ready classifier with an average inference latency of ~21ms on CPU.

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

After placing the trained model in `./intent_model/`, the Flask application serves the model for real-time inference:

```bash
python app.py
```

The server runs at `http://127.0.0.1:5000` and accepts:

- **Plain text** (JSON body or form field)
- **File uploads**: `.txt`, `.eml`, `.pdf`, `.docx`

The verdict system operates on a **three-tier confidence threshold** (60%):

| Confidence | Verdict | Response |
|---|---|---|
| ≥ 60%, phishing label | `Phishing Attempt Detected` | Top-1 intent + confidence |
| ≥ 60%, Safe/Neutral label | `Message Appears Safe` | Top-1 intent + confidence |
| < 60%, either direction | `Could Be a Scam` | Top-3 intent signals |

**Confident phishing response:**
```json
{
  "verdict": "Phishing Attempt Detected",
  "is_phishing": true,
  "intent": "Authority",
  "confidence": "94.31"
}
```

**Uncertain response:**
```json
{
  "verdict": "Could Be a Scam",
  "is_phishing": null,
  "intent": "Authority",
  "confidence": "45.23",
  "top_intents": [
    {"label": "Authority", "confidence": "45.23"},
    {"label": "Fear", "confidence": "32.11"},
    {"label": "Safe / Neutral", "confidence": "18.50"}
  ]
}
```

---

## Model Artifact

The trained model is hosted on HuggingFace Hub. To use it locally, download and place it in `./intent_model/`:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="your-username/phishing-intent-classifier", local_dir="./intent_model")
```

| File | Description |
|---|---|
| `model.safetensors` | Fine-tuned DistilBERT weights (~267MB) |
| `config.json` | Architecture config including `id2label` and `label2id` mappings for all 7 classes |
| `tokenizer.json` | Self-contained fast tokenizer (HuggingFace unified format) |
| `tokenizer_config.json` | Tokenizer runtime settings |
| `training_args.bin` | Snapshot of the training configuration used to produce this checkpoint |

---

## Benchmarks

Full results including per-class F1 scores, latency measurements, and per-prediction analysis are documented in [`RESULTS.md`](./RESULTS.md).

| Metric | Value |
|---|---|
| Validation accuracy | 81.92% |
| Validation macro_f1 | 59.91% |
| Inference latency (CPU) | ~21ms/sequence |
| Throughput (CPU) | ~47 sequences/sec |

---

## Dataset

Source: [`rokibulroni/Phishing-Email-Dataset`](https://huggingface.co/datasets/rokibulroni/Phishing-Email-Dataset)

| Split | Count |
|---|---|
| Phishing | ~42,256 |
| Safe / Neutral | ~39,510 |
| Total (after deduplication) | ~81,766 |
