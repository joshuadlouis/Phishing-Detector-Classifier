import time
import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

def run_benchmarks(model_path="./intent_model"):
    print(f"Loading model and tokenizer from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # DistilBERT has no segment embeddings — suppress token_type_ids
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Device selection: CUDA > DirectML (AMD) > CPU
    device = -1
    if torch.cuda.is_available():
        device = 0
    else:
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
        except ImportError:
            pass

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    # -------------------------------------------------------------------------
    # Golden test set — 2 examples per label (14 total, no Liking)
    # -------------------------------------------------------------------------
    golden_tests = [
        {"text": "Act quickly to secure your account before midnight, time is running out!", "true_label": "Urgency and Scarcity"},
        {"text": "Only 3 spots left. Your access will be revoked if you don't respond within the hour.", "true_label": "Urgency and Scarcity"},
        {"text": "This is FBI Director Wray. Your computer is flagged for illegal activity. Comply immediately.", "true_label": "Authority"},
        {"text": "IT Helpdesk: You must change your password as mandated by the University President.", "true_label": "Authority"},
        {"text": "Your account has been hacked and all your files will be leaked unless you pay $500.", "true_label": "Fear"},
        {"text": "Warning: Failure to verify will result in irreversible data loss and permanent account closure.", "true_label": "Fear"},
        {"text": "You have been selected to receive a completely free $10,000 lottery payout! Click here to claim.", "true_label": "Greed and Reciprocity"},
        {"text": "Since we did you a favour last time, please wire the $5,000 bonus to our associate.", "true_label": "Greed and Reciprocity"},
        {"text": "As we discussed previously, please continue with the payment schedule we agreed upon yesterday.", "true_label": "Commitment and Consistency"},
        {"text": "Please finalize the remaining verification step you started earlier today.", "true_label": "Commitment and Consistency"},
        {"text": "All of your co-workers have already signed up for the new benefits portal. Don't be left behind!", "true_label": "Consensus and Social Proof"},
        {"text": "98% of customers have already updated their profiles. Join them today.", "true_label": "Consensus and Social Proof"},
        {"text": "Hi team, just sharing the notes from last Thursday's all-hands meeting. No action required.", "true_label": "Safe / Neutral"},
        {"text": "Your invoice for March has been attached. Please review at your earliest convenience.", "true_label": "Safe / Neutral"},
    ]

    texts      = [item["text"] for item in golden_tests]
    true_labels = [item["true_label"] for item in golden_tests]

    print(f"\n--- Running Benchmarks on {len(texts)} samples ---")

    # -------------------------------------------------------------------------
    # 1. Latency benchmark
    # -------------------------------------------------------------------------
    # Warm-up pass (prevents cold-start skewing first-run latency)
    _ = classifier(texts[0], truncation=True, max_length=256)

    start_time = time.time()
    results    = classifier(texts, truncation=True, max_length=256)
    end_time   = time.time()

    total_time      = end_time - start_time
    avg_latency_ms  = (total_time / len(texts)) * 1000

    print(f"\nTotal Inference Time : {total_time:.4f}s")
    print(f"Average Latency      : {avg_latency_ms:.2f} ms/sequence")
    print(f"Throughput           : {len(texts) / total_time:.2f} sequences/sec")

    # -------------------------------------------------------------------------
    # 2. Accuracy & classification report
    # -------------------------------------------------------------------------
    predicted_labels = [res['label'] for res in results]
    model_labels     = list(model.config.id2label.values())

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n--- Accuracy ---")
    print(f"Golden-set accuracy: {accuracy * 100:.2f}%")

    print("\n--- Per-Class Report ---")
    print(classification_report(
        true_labels, predicted_labels,
        labels=model_labels,
        zero_division=0
    ))

    # -------------------------------------------------------------------------
    # 3. Prediction vs ground truth table
    # -------------------------------------------------------------------------
    print("--- Predictions vs Ground Truth ---")
    df = pd.DataFrame({
        "Text snippet"       : [t[:50] + "..." for t in texts],
        "Expected"           : true_labels,
        "Predicted"          : predicted_labels,
        "Confidence"         : [f"{res['score']*100:.1f}%" for res in results],
        "Correct"            : ["PASS" if t == p else "FAIL" for t, p in zip(true_labels, predicted_labels)],
    })
    pd.set_option('display.max_colwidth', None)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmarks()
