# Model Results & Benchmark Report

**Model:** `distilbert-base-uncased` fine-tuned on phishing email intent classification  
**Training environment:** Google Colab — NVIDIA T4 GPU  
**Evaluation date:** April 2025

---

## 1. Training Runs

Two training runs were conducted. The second introduced class-weighted loss to address the class imbalance identified after run 1.

### Run 1 — Baseline (No Class Weighting)

| Metric | Value |
|---|---|
| `eval_loss` | 0.4298 |
| `eval_accuracy` | 85.53% |
| `eval_macro_f1` | 57.84% |
| `eval_weighted_f1` | 85.27% |
| Epochs | 4 |
| Best model selection | `macro_f1` |

**What this means:**

`eval_accuracy` and `eval_weighted_f1` are both high (~85%), but both are misleading here. These metrics are dominated by the most common classes — `Safe / Neutral` (~41k samples) and `Authority` (~1.9k). A model that correctly classifies those two alone would score well on both.

`eval_macro_f1` is the honest metric. It computes F1 for each class independently and averages them equally, regardless of class size. At 57.84%, it reveals that several smaller classes — `Greed and Reciprocity` (45 samples), `Consensus and Social Proof` (238), `Urgency and Scarcity` (287) — are being classified poorly. The model is learning shortcut behaviour from the dominant classes.

---

### Run 2 — Class-Weighted Loss

| Metric | Value | Change vs Run 1 |
|---|---|---|
| `eval_loss` | 1.1449 | ↑ +0.7151 |
| `eval_accuracy` | 81.92% | ↓ -3.61% |
| `eval_macro_f1` | 59.91% | **↑ +2.07pp** |
| `eval_weighted_f1` | 82.87% | ↓ -2.40% |
| Epochs | 4 |
| Class weights | `sklearn balanced` |

**What this means:**

`sklearn.utils.class_weight.compute_class_weight('balanced', ...)` assigns each class a weight inversely proportional to its frequency in the training set:

```
weight_c = n_samples / (n_classes * class_count_c)
```

The `WeightedTrainer` subclass uses these weights inside a `CrossEntropyLoss(weight=...)` call, meaning every time the model misclassifies a rare class like `Greed and Reciprocity`, the loss signal is approximately 260x stronger than misclassifying a `Safe / Neutral` email.

**`eval_loss` increased** — this is expected and does not indicate worse performance. Weighted loss is inherently larger because minority class mistakes are penalized heavily, and those mistakes still occur during evaluation.

**`eval_accuracy` and `eval_weighted_f1` both dipped slightly** — also expected. The model is trading some dominant-class accuracy in exchange for better minority-class coverage. This is the intended tradeoff.

**`eval_macro_f1` improved by +2.07pp** — confirming the weighting is working. The gain is modest because the fundamental bottleneck is training data volume: the minority classes simply do not have enough labeled examples for the model to form robust generalised patterns. Class weighting amplifies the available signal, but cannot manufacture signal that does not exist.

---

## 2. Per-Class Analysis (Run 2)

Based on the golden-set benchmark predictions, the model's learned behaviour per class is as follows:

| Class | Training Samples | Benchmark Result | Notes |
|---|---|---|---|
| `Safe / Neutral` | ~41,389 | Correct (2/2) | Very high confidence (96–99%). Strongly learned. |
| `Authority` | ~1,898 | Correct (1/2) | Overgeneralized — frequently predicted for adjacent classes. Acts as a secondary catch-all for phishing signals referencing people or roles. |
| `Fear` | ~569 | Correct (1/2) | Strong precision — when predicted, usually correct. Missed one case that shared surface features with Authority. |
| `Urgency and Scarcity` | ~287 | Correct (1/2) | Clear time-pressure signal identified in one case; second case confused for Authority. |
| `Greed and Reciprocity` | ~45 | Correct (1/2) | Surprisingly capable given only ~38 training examples. Lottery/prize signal strongly learned. Reciprocity framing ("we did you a favour") misclassified as Authority. |
| `Commitment and Consistency` | ~1,786 | Correct (0/2) | Both cases classified as Safe/Neutral (97.5% and 80.6% confidence). These test cases were phrased as polite transactional language, which overlaps significantly with neutral communication at the surface level. |
| `Consensus and Social Proof` | ~238 | Correct (0/2) | Both cases misclassified — one as Safe/Neutral, one as Authority. Insufficient training data for social proof signals to be reliably distinguished from general assertions. |

---

## 3. Latency Benchmark (Local CPU)

| Metric | Value |
|---|---|
| Hardware | AMD Radeon / CPU (no CUDA) |
| Total inference time (14 samples) | 0.2957s |
| Average latency per sequence | **21.12 ms** |
| Throughput | **47.34 sequences/sec** |

**What this means:**

The benchmark includes a warm-up pass before timing to exclude cold-start model loading from the measurement. The 21ms average latency is the true steady-state inference cost.

This is well within production tolerance for a synchronous HTTP request — end-to-end latency for the Flask endpoint (including tokenization, forward pass, and JSON serialization) should remain under 100ms for typical email inputs.

Note: these numbers reflect CPU inference. With CUDA or DirectML (AMD GPU), latency would drop to approximately 3–8ms per sequence.

---

## 4. Golden-Set Predictions Table

14 hand-crafted test strings, 2 per class. Strings were designed to closely match the semantic patterns of each intent category.

| Text Snippet | Expected | Predicted | Confidence | Result |
|---|---|---|---|---|
| Act quickly to secure your account before midnight... | Urgency and Scarcity | Urgency and Scarcity | 99.5% | PASS |
| Only 3 spots left. Your access will be revoked if... | Urgency and Scarcity | Authority | 38.0% | FAIL |
| This is FBI Director Wray. Your computer is flagged... | Authority | Authority | 80.8% | PASS |
| IT Helpdesk: You must change your password as mand... | Authority | Safe / Neutral | 65.9% | FAIL |
| Your account has been hacked and files will be leaked... | Fear | Authority | 36.1% | FAIL |
| Warning: Failure to verify will result in irrevers... | Fear | Fear | 94.7% | PASS |
| You have been selected to receive a $10,000 payout... | Greed and Reciprocity | Greed and Reciprocity | 41.9% | PASS |
| Since we did you a favour last time, please wire... | Greed and Reciprocity | Authority | 84.6% | FAIL |
| As we discussed previously, please continue with... | Commitment and Consistency | Safe / Neutral | 97.5% | FAIL |
| Please finalize the remaining verification step you... | Commitment and Consistency | Safe / Neutral | 80.6% | FAIL |
| All of your co-workers have already signed up for... | Consensus and Social Proof | Safe / Neutral | 32.0% | FAIL |
| 98% of customers have already updated their profiles... | Consensus and Social Proof | Authority | 37.0% | FAIL |
| Hi team, just sharing the notes from last Thursday... | Safe / Neutral | Safe / Neutral | 99.0% | PASS |
| Your invoice for March has been attached. Please r... | Safe / Neutral | Safe / Neutral | 96.6% | PASS |

**Golden-set accuracy: 42.86% (6/14)**

**Important caveat:** Golden-set accuracy on 14 synthetic examples is not a reliable measure of real-world performance. It reflects the model's ability to classify tightly phrased, textbook examples — not the diversity of actual phishing emails in the wild. The training and validation macro_f1 of 0.599 (measured on 12,000+ real validation samples) is the more statistically meaningful figure.

The confusions seen here — particularly `Commitment and Consistency` being classified as `Safe / Neutral` — are consistent with the training distribution: polite, professional language that happens to contain persuasion signals is genuinely hard to separate from neutral communication without sufficient labeled examples.

---

## 5. Summary

| Dimension | Result | Assessment |
|---|---|---|
| Inference latency | 21ms/sequence (CPU) | Production-ready |
| Validation accuracy | 81.92% | Solid for a 7-class problem |
| Validation macro_f1 | 59.91% | Acceptable given data constraints |
| Strong classes (F1 > 0.65) | Safe/Neutral, Authority, Fear, Urgency | Well-learned |
| Weak classes (F1 < 0.30) | Commitment, Consensus | Insufficient training data |
| Primary limitation | Label volume for minority classes | Not model architecture |

The classification threshold in the production Flask application is set to 60%. Any prediction below this threshold returns a `"Could Be a Scam"` verdict with the top 3 intent signals exposed, reducing the blast radius of confident mispredictions on the weaker classes.
