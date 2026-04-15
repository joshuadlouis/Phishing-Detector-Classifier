from flask import Flask, request, render_template, jsonify
from bs4 import BeautifulSoup
import torch
import io
import email
from email import policy
import PyPDF2
import docx
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load intent model locally
MODEL_PATH = "./intent_model"
print(f"Loading intent model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# DistilBERT has no segment embeddings — suppress token_type_ids the tokenizer
# would otherwise produce (inherited from its BERT-style vocabulary).
tokenizer.model_input_names = ["input_ids", "attention_mask"]
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
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

def clean_text(text):
    """Strips HTML tags and standardizes text."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return text.strip()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    raw_text = ""
    
    # Handle file uploads if sent via multipart/form-data
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            ext = file.filename.split('.')[-1].lower()
            try:
                if ext == "pdf":
                    pdf_reader = PyPDF2.PdfReader(file)
                    raw_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                elif ext == "docx":
                    doc = docx.Document(file)
                    raw_text = " ".join([para.text for para in doc.paragraphs])
                elif ext == "eml":
                    msg = email.message_from_bytes(file.read(), policy=policy.default)
                    body = msg.get_body(preferencelist=('plain', 'html'))
                    if body:
                        raw_text = body.get_content()
                    else:
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() in ["text/plain", "text/html"]:
                                    raw_text += part.get_payload(decode=True).decode(errors='ignore')
                        else:
                            raw_text = msg.get_payload(decode=True).decode(errors='ignore')
                elif ext == "txt":
                    raw_text = file.read().decode("utf-8", errors="ignore")
                else:
                    return jsonify({"error": f"Unsupported file extension: .{ext}"}), 400
            except Exception as e:
                return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400
    
    # If no file or empty file, check JSON body or form text
    if not raw_text:
        if request.is_json:
            raw_text = request.json.get("text", "")
        else:
            raw_text = request.form.get("text", "")
            
    if not raw_text:
        return jsonify({"error": "No text or valid file provided"}), 400
        
    cleaned_text = clean_text(raw_text)
    
    if not cleaned_text:
        return jsonify({"error": "Text was empty after cleaning"}), 400
        
    try:
        # Confidence threshold — below this, verdict becomes "Could Be a Scam"
        # and the top 3 intents are returned instead of just the top 1.
        CONFIDENCE_THRESHOLD = 0.60

        result = classifier(cleaned_text, truncation=True, max_length=512, top_k=3)
        top     = result[0]
        intent  = top['label']
        score   = top['score']

        if score >= CONFIDENCE_THRESHOLD:
            # Confident prediction — single clear verdict
            is_phishing  = intent != "Safe / Neutral"
            verdict_text = "Phishing Attempt Detected" if is_phishing else "Message Appears Safe"
            return jsonify({
                "verdict":     verdict_text,
                "is_phishing": is_phishing,
                "intent":      intent,
                "confidence":  f"{score * 100:.2f}",
            })
        else:
            # Model is uncertain — surface top 3 intents and flag as ambiguous
            top_intents = [
                {"label": r['label'], "confidence": f"{r['score'] * 100:.2f}"}
                for r in result
            ]
            return jsonify({
                "verdict":     "Could Be a Scam",
                "is_phishing": None,
                "intent":      intent,
                "confidence":  f"{score * 100:.2f}",
                "top_intents": top_intents,
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
