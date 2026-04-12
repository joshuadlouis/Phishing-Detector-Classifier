import time
import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

def run_benchmarks(model_path="./intent_model"):
    print(f"Loading model and tokenizer from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Use GPU for inference if available
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    
    # 1. Define Golden Test Set
    # A curated list of synthetic strings mapping tightly to our 7 trained Intents
    golden_tests = [
        {"text": "Act quickly to secure your account before midnight, time is running out!", "true_label": "Urgency and Scarcity"},
        {"text": "Only 3 items left in stock. We will delete your account today if you don't respond.", "true_label": "Urgency and Scarcity"},
        {"text": "This is FBI Director Wray. Your computer is flagged for illegal activity. Comply immediately.", "true_label": "Authority"},
        {"text": "IT Helpdesk: You must change your password as mandated by the University President.", "true_label": "Authority"},
        {"text": "Your account has been hacked and all your files will be leaked unless you pay $500.", "true_label": "Fear"},
        {"text": "Warning: Failure to verify will result in irreversible data loss and account permanent closure.", "true_label": "Fear"},
        {"text": "You have been selected to receive a completely free $10,000 lottery payout! Click here to claim.", "true_label": "Greed and Reciprocity"},
        {"text": "Since we did you a favor last time, please wire the $5000 bonus to my associate.", "true_label": "Greed and Reciprocity"},
        {"text": "As we discussed previously, please continue with the payment schedule we agreed upon yesterday.", "true_label": "Commitment and Consistency"},
        {"text": "Please finalize the remaining verification step as you started earlier today.", "true_label": "Commitment and Consistency"},
        {"text": "All of your co-workers have already signed up for the new benefits portal. Don't be left behind!", "true_label": "Consensus and Social Proof"},
        {"text": "98% of customers are updating their profiles today.", "true_label": "Consensus and Social Proof"},
        {"text": "Hey friend! Hope you're doing great. Could you quickly check out this document for me? Thanks buddy!", "true_label": "Liking"},
        {"text": "Hi Jane, it's so great to reconnect! By the way, check out this link when you can.", "true_label": "Liking"}
    ]
    
    texts = [item["text"] for item in golden_tests]
    true_labels = [item["true_label"] for item in golden_tests]
    
    print(f"\n--- Running Benchmarks on {len(texts)} samples ---")
    
    # 2. Benchmarking Inference Speed & Latency
    start_time = time.time()
    
    # Run Inference
    results = classifier(texts, truncation=True, max_length=512)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_latency_ms = (total_time / len(texts)) * 1000
    
    print(f"Total Inference Time: {total_time:.4f} seconds")
    print(f"Average Latency:      {avg_latency_ms:.2f} ms per sequence")
    print(f"Throughput:           {len(texts) / total_time:.2f} sequence/sec\n")
    
    # 3. Computing Accuracy
    predicted_labels = [res['label'] for res in results]
    
    # Get all 7 possible labels mapped in model config
    model_labels = list(model.config.id2label.values())
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"--- Global Accuracy ---")
    print(f"Model Correctness: {accuracy * 100:.2f}% (Note: Expected to be low if trained on very few samples!)\n")
    
    print("--- Detailed Classification Report ---")
    report = classification_report(true_labels, predicted_labels, labels=model_labels, zero_division=0)
    print(report)
        
    print("\n--- Predictions vs. Ground Truth ---")
    df = pd.DataFrame({
        "Sample Text snippet": [t[:45] + "..." for t in texts],
        "Expected (Ground Truth)": true_labels,
        "Actual Predicted": predicted_labels,
        "Match": [t == p for t, p in zip(true_labels, predicted_labels)]
    })
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmarks()
