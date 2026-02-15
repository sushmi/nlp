import sys
import os

# Add the code directory to path so we can import the BERT module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import torch
import torch.nn as nn
from transformers import BertTokenizer
from flask import Flask, render_template_string, request

from bert import BERT

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
BERT_CHECKPOINT = os.path.join(MODEL_DIR, 'sentence_bert_checkpoint.pt')
CLASSIFIER_CHECKPOINT = os.path.join(MODEL_DIR, 'classifier_head.pt')
MAX_SEQ_LENGTH = 128
NLI_LABELS = ["Entailment", "Neutral", "Contradiction"]

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

# --- Load model and tokenizer ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load BERT
params, state_dict = torch.load(BERT_CHECKPOINT, map_location=device)
model = BERT(**params, device=device).to(device)
model.load_state_dict(state_dict)
model.eval()

# Load classifier head
classifier_head = nn.Linear(params['d_model'] * 3, 3).to(device)
classifier_head.load_state_dict(torch.load(CLASSIFIER_CHECKPOINT, map_location=device))
classifier_head.eval()

print(f"Model loaded on {device}")


def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


def predict_nli(premise: str, hypothesis: str):
    """Predict NLI label and probabilities for a premise-hypothesis pair."""
    inputs_a = tokenizer(premise, return_tensors='pt', max_length=MAX_SEQ_LENGTH,
                         truncation=True, padding='max_length').to(device)
    inputs_b = tokenizer(hypothesis, return_tensors='pt', max_length=MAX_SEQ_LENGTH,
                         truncation=True, padding='max_length').to(device)

    segment_ids = torch.zeros(1, MAX_SEQ_LENGTH, dtype=torch.int32).to(device)

    with torch.no_grad():
        u = model.get_last_hidden_state(inputs_a['input_ids'], segment_ids)
        v = model.get_last_hidden_state(inputs_b['input_ids'], segment_ids)

        u_pool = mean_pool(u, inputs_a['attention_mask'])
        v_pool = mean_pool(v, inputs_b['attention_mask'])

        uv_abs = torch.abs(u_pool - v_pool)
        x = torch.cat([u_pool, v_pool, uv_abs], dim=-1)

        logits = classifier_head(x)
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
        pred = torch.argmax(logits, dim=-1).item()

    return NLI_LABELS[pred], {NLI_LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs)}


# --- Flask App ---
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLI Prediction - Sentence BERT</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f7fa; min-height: 100vh; display: flex; align-items: center;
               justify-content: center; padding: 20px; }
        .container { background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                     padding: 40px; max-width: 700px; width: 100%; }
        h1 { text-align: center; color: #1a1a2e; margin-bottom: 8px; font-size: 1.6em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 0.95em; }
        label { display: block; font-weight: 600; color: #333; margin-bottom: 6px; font-size: 0.95em; }
        textarea { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px;
                   font-size: 1em; resize: vertical; min-height: 70px; transition: border-color 0.2s;
                   font-family: inherit; }
        textarea:focus { outline: none; border-color: #4a90d9; }
        .field { margin-bottom: 20px; }
        button { width: 100%; padding: 14px; background: #4a90d9; color: white; border: none;
                 border-radius: 8px; font-size: 1.05em; font-weight: 600; cursor: pointer;
                 transition: background 0.2s; }
        button:hover { background: #357abd; }
        .result { margin-top: 28px; padding: 24px; background: #f8f9fb; border-radius: 10px;
                  border: 1px solid #e8e8e8; }
        .result h2 { font-size: 1.1em; color: #333; margin-bottom: 16px; }
        .label-tag { display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: 700;
                     font-size: 1.1em; margin-bottom: 16px; }
        .label-entailment { background: #d4edda; color: #155724; }
        .label-neutral { background: #fff3cd; color: #856404; }
        .label-contradiction { background: #f8d7da; color: #721c24; }
        .prob-bar-container { margin-bottom: 8px; }
        .prob-label { font-size: 0.9em; color: #555; margin-bottom: 3px; }
        .prob-bar { height: 22px; border-radius: 4px; background: #e9ecef; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 4px; display: flex; align-items: center;
                     padding-left: 8px; font-size: 0.8em; color: white; font-weight: 600;
                     min-width: 40px; transition: width 0.4s ease; }
        .fill-entailment { background: #28a745; }
        .fill-neutral { background: #ffc107; color: #333; }
        .fill-contradiction { background: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Natural Language Inference</h1>
        <p class="subtitle">Predict the relationship between two sentences using a custom-trained Sentence BERT model</p>
        <form method="POST">
            <div class="field">
                <label for="premise">Premise</label>
                <textarea id="premise" name="premise" placeholder="e.g., A man is playing a guitar on stage.">{{ premise }}</textarea>
            </div>
            <div class="field">
                <label for="hypothesis">Hypothesis</label>
                <textarea id="hypothesis" name="hypothesis" placeholder="e.g., The man is performing music.">{{ hypothesis }}</textarea>
            </div>
            <button type="submit">Predict</button>
        </form>

        {% if result %}
        <div class="result">
            <h2>Prediction</h2>
            <span class="label-tag label-{{ result.label | lower }}">{{ result.label }}</span>
            <div>
                {% for label, prob in result.probs.items() %}
                <div class="prob-bar-container">
                    <div class="prob-label">{{ label }}: {{ prob }}%</div>
                    <div class="prob-bar">
                        <div class="prob-fill fill-{{ label | lower }}" style="width: {{ prob }}%">
                            {% if prob > 5 %}{{ prob }}%{% endif %}
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    premise = ""
    hypothesis = ""
    result = None

    if request.method == "POST":
        premise = request.form.get("premise", "").strip()
        hypothesis = request.form.get("hypothesis", "").strip()

        if premise and hypothesis:
            label, probs = predict_nli(premise, hypothesis)
            result = {"label": label, "probs": probs}

    return render_template_string(HTML_TEMPLATE, premise=premise,
                                  hypothesis=hypothesis, result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
