"""
A6 evaluation — generates A6/answer/response-st-xxxxxx-chapter-6.json

Retriever : TF-IDF + cosine similarity  (sklearn, no GPU needed)
Generator : Qwen/Qwen2.5-1.5B-Instruct  (CPU, local)
Contextual: rule-based section-header enrichment

Run: uv run python A6/code/generate_results.py
"""

# Disable ALL GPU backends before any torch import
import os, re, json, pickle
os.environ["CUDA_VISIBLE_DEVICES"]       = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE     = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE, "../data/6.pdf")
OUT_PATH = os.path.join(BASE, "../answer/response-st-xxxxxx-chapter-6.json")
TITLE    = "Chapter 6: Neural Networks"
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

load_dotenv(os.path.join(BASE, "../../.env"))
os.makedirs(os.path.join(BASE, "../answer"), exist_ok=True)


# ── Cleaning ──────────────────────────────────────────────────────────────────
def clean_page(text: str) -> str:
    for ch, rep in {"ﬁ":"fi","ﬀ":"ff","ﬂ":"fl","\ufb01":"fi","\ufb02":"fl",
                    "\ufb00":"ff","\ufb03":"ffi","\ufb04":"ffl"}.items():
        text = text.replace(ch, rep)
    text = re.sub(r"(\w+)-\n(\w)", r"\1\2", text)
    text = re.sub(r"^\d{1,3}\n+CHAPTER\s+\d+\s*\n+[•·]\s*\n+[A-Z ,]+\n+", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\d{1,3}\s*$", "", text)
    text = re.sub(r"(?m)^(Figure|Table)\s+\d+\.\d+.*$", "", text)
    text = re.sub(r"(?m)^\(\d+\.\d+\)\s*$", "", text)
    cleaned = []
    for line in text.split("\n"):
        s = line.strip(); words = s.split()
        if (1<=len(words)<=3 and s==s.lower() and s[-1:] not in ".?!:,"
                and not re.search(r"\d", s) and not s.startswith("•")):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ── Section map ───────────────────────────────────────────────────────────────
SECTION_RE = re.compile(r"^(6\.\d+(?:\.\d+)?)\n([A-Z][^\n]{1,60})$", re.MULTILINE)

def build_section_map(full_text):
    return [(m.start(), m.group(1), m.group(2).strip()) for m in SECTION_RE.finditer(full_text)]

def get_section(chunk, full_text, section_map):
    pos = full_text.find(chunk[:80])
    prev = [(o,n,t) for o,n,t in section_map if o <= pos] if pos != -1 else []
    return f"Section {prev[-1][1]}: {prev[-1][2]}" if prev else TITLE


# ── Load & chunk PDF ──────────────────────────────────────────────────────────
print("Loading PDF…")
docs = [Document(page_content=clean_page(d.page_content), metadata=d.metadata)
        for d in PyMuPDFLoader(PDF_PATH).load()]
full_text   = "\n\n".join(d.page_content for d in docs)
section_map = build_section_map(full_text)
print(f"  {len(docs)} pages | {len(section_map)} sections")

splitter    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunk_texts = [c.page_content for c in splitter.split_documents(docs)]
print(f"  {len(chunk_texts)} chunks")


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────
print("Building TF-IDF indices…")
# Naive: plain chunks
naive_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
naive_mat = naive_vec.fit_transform(chunk_texts)

# Contextual: section-enriched chunks
enriched_texts = [
    f"This chunk from {TITLE} discusses content from {get_section(c, full_text, section_map)}.\n\n{c}"
    for c in chunk_texts
]
ctx_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
ctx_mat = ctx_vec.fit_transform(enriched_texts)
print("  Done.")


def retrieve_naive(question, k=3):
    q   = naive_vec.transform([question])
    sim = cosine_similarity(q, naive_mat)[0]
    idx = np.argsort(sim)[::-1][:k]
    return [chunk_texts[i] for i in idx]

def retrieve_ctx(question, k=3):
    q   = ctx_vec.transform([question])
    sim = cosine_similarity(q, ctx_mat)[0]
    idx = np.argsort(sim)[::-1][:k]
    return [enriched_texts[i] for i in idx]


# ── Load Qwen generator (CPU) ─────────────────────────────────────────────────
print(f"\nLoading {GEN_MODEL} on CPU…")
import torch
torch.set_num_threads(4)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float32,
                                                  low_cpu_mem_usage=True)
gen_model.eval()
print("  Ready.")


def generate(context: str, question: str, max_new_tokens: int = 180) -> str:
    messages = [
        {"role": "system", "content":
            "Answer using ONLY the given context. Be concise and accurate."},
        {"role": "user", "content": f"Context:\n{context[:1500]}\n\nQuestion: {question}"},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def naive_rag(question):
    return generate("\n\n---\n\n".join(retrieve_naive(question)), question)

def contextual_rag(question):
    return generate("\n\n---\n\n".join(retrieve_ctx(question)), question)


# ── QA pairs ─────────────────────────────────────────────────────────────────
qa_pairs = [
    {"question": "What is a neural unit and what computation does it perform?",
     "ground_truth_answer": "A neural unit is the building block of a neural network. It takes a set of real-valued inputs, computes a weighted sum plus a bias term z = w·x + b, applies a non-linear activation function f, and produces output y = f(z)."},
    {"question": "What is the bias term in a neural unit and why is it needed?",
     "ground_truth_answer": "The bias term b is an additional scalar added to the weighted sum of inputs. It allows the unit to shift its activation threshold regardless of the input values, giving the network greater flexibility."},
    {"question": "What is the sigmoid activation function and what is its output range?",
     "ground_truth_answer": "The sigmoid function is σ(z) = 1/(1+e^{-z}). It maps any real value to the range (0, 1), squashing extreme values toward 0 or 1. It is differentiable, which is useful for gradient-based learning."},
    {"question": "What are the three common non-linear activation functions used in neural networks?",
     "ground_truth_answer": "The three common non-linear activation functions are sigmoid (σ), tanh (hyperbolic tangent), and ReLU (Rectified Linear Unit). Each maps the weighted sum z to a different output range and has different properties useful in practice."},
    {"question": "What is the ReLU activation function and how is it defined?",
     "ground_truth_answer": "ReLU (Rectified Linear Unit) is defined as ReLU(z) = max(0, z). It returns 0 for negative inputs and the input value itself for positive inputs. It is computationally efficient and avoids the vanishing gradient problem for positive values."},
    {"question": "What is the tanh activation function and how does it differ from sigmoid?",
     "ground_truth_answer": "The tanh function maps inputs to the range (-1, 1) using tanh(z) = (e^z − e^{-z})/(e^z + e^{-z}). Unlike sigmoid which outputs (0,1), tanh is zero-centered, which can speed up learning since outputs are symmetric around zero."},
    {"question": "What is the XOR problem and why is it significant for neural networks?",
     "ground_truth_answer": "XOR is not linearly separable — no single straight line can separate its two classes. A single-layer perceptron cannot solve XOR. This demonstrates the need for hidden layers with non-linear activation functions, which allow networks to learn non-linear decision boundaries."},
    {"question": "What is a feedforward neural network?",
     "ground_truth_answer": "A feedforward neural network is a network where units are organised in layers — input, one or more hidden layers, and an output layer — and connections flow only forward (no cycles). Each layer applies a linear transformation followed by a non-linear activation."},
    {"question": "What is backpropagation and what role does it play in training?",
     "ground_truth_answer": "Backpropagation is an algorithm for computing the gradient of the loss with respect to every weight in the network. It applies the chain rule to propagate error signals from the output layer backward through each layer, enabling gradient descent to update all weights efficiently."},
    {"question": "What is gradient descent and how are weights updated during training?",
     "ground_truth_answer": "Gradient descent is an optimisation algorithm that iteratively moves weights in the direction that reduces the loss. Each weight is updated as w ← w − η·(∂L/∂w), where η is the learning rate and ∂L/∂w is the gradient of the loss with respect to that weight."},
    {"question": "What is the learning rate and why does its choice matter?",
     "ground_truth_answer": "The learning rate η is a hyperparameter controlling the step size during gradient descent. A learning rate that is too small leads to slow convergence, while one that is too large can cause the loss to oscillate or diverge, making the choice of learning rate critical to training stability."},
    {"question": "What is dropout regularisation and why is it used?",
     "ground_truth_answer": "Dropout is a regularisation technique where, during training, each unit is independently set to zero with probability p. This prevents co-adaptation of neurons and reduces overfitting by forcing the network to learn more robust features."},
    {"question": "What is the softmax function and when is it used in neural networks?",
     "ground_truth_answer": "Softmax normalises a vector of real values into a probability distribution that sums to 1: softmax(z_i) = e^{z_i}/Σ_j e^{z_j}. It is used in the output layer for multi-class classification, turning raw scores (logits) into class probabilities."},
    {"question": "What is cross-entropy loss and why is it preferred for classification?",
     "ground_truth_answer": "Cross-entropy loss measures the dissimilarity between the predicted probability distribution and the true label distribution: L = −Σ y_i log(ŷ_i). It is preferred for classification because it penalises confident wrong predictions heavily and its gradient with softmax is simple (ŷ − y)."},
    {"question": "What is a computation graph and how is it used in neural network training?",
     "ground_truth_answer": "A computation graph is a directed acyclic graph representing the sequence of operations that compute the output from the input. During backpropagation, the graph is traversed in reverse to efficiently compute gradients using the chain rule at each node."},
    {"question": "What is the vanishing gradient problem and which activation functions suffer from it?",
     "ground_truth_answer": "The vanishing gradient problem occurs when gradients become extremely small as they are backpropagated through many layers, making early-layer weights update very slowly. Sigmoid and tanh suffer from this because their gradients approach zero in their saturation regions, unlike ReLU."},
    {"question": "How does L2 regularisation reduce overfitting in neural networks?",
     "ground_truth_answer": "L2 regularisation adds a penalty term λ‖w‖² to the loss function that discourages large weight values. This shrinks weights toward zero during gradient descent, reducing the model's complexity and improving generalisation to unseen data."},
    {"question": "What is mini-batch stochastic gradient descent (SGD)?",
     "ground_truth_answer": "Mini-batch SGD updates weights using the gradient computed over a small random subset (mini-batch) of training examples rather than the full dataset (batch) or a single example (online). It balances computational efficiency with gradient noise, which can help escape local minima."},
    {"question": "What is the difference between the input layer, hidden layers, and the output layer in a feedforward network?",
     "ground_truth_answer": "The input layer receives the raw feature vectors. Hidden layers apply learned linear transformations followed by non-linear activations to extract increasingly abstract representations. The output layer produces the final prediction — a class probability via softmax for classification or a continuous value for regression."},
    {"question": "Why are non-linear activation functions necessary in multi-layer neural networks?",
     "ground_truth_answer": "Without non-linear activations, stacking multiple linear layers is equivalent to a single linear transformation, providing no additional expressive power. Non-linear activation functions allow the network to learn non-linear decision boundaries and represent complex functions such as XOR."},
]

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nGenerating answers for {len(qa_pairs)} QA pairs…")
results = []
for i, qa in enumerate(qa_pairs):
    q = qa["question"]
    print(f"  [{i+1:02d}/{len(qa_pairs)}] {q[:65]}…", flush=True)
    results.append({
        "question":                    q,
        "ground_truth_answer":         qa["ground_truth_answer"],
        "naive_rag_answer":            naive_rag(q),
        "contextual_retrieval_answer": contextual_rag(q),
    })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(results)} results → {OUT_PATH}")
