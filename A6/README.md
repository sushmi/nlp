# A6 — Naive RAG vs Contextual Retrieval

A domain-specific QA system built on **Chapter 6: Neural Networks** (Jurafsky & Martin, *Speech and Language Processing*, 3rd ed.).
The assignment compares two retrieval strategies — Naive RAG and Contextual Retrieval — and wraps the best pipeline in a simple web chatbot.

---

## Project Structure

```
A6/
├── code/
│   ├── a6.ipynb              # Main notebook (Tasks 1 & 2)
│   └── generate_results.py   # Standalone script to produce the answer JSON
├── app/
│   └── app.py                # Streamlit chatbot (Task 3)
├── data/
│   └── 6.pdf                 # Source chapter (Jurafsky & Martin Ch. 6)
├── answer/
│   └── response-st126526-chapter-6.json   # Evaluation output (Task 3.4)
└── models/
    └── enriched_chunks.pkl   # Cached enriched chunks (auto-generated)
```

---

## Task 1 — Source Discovery & Data Preparation

### 1.1 Chapter Selection

Last digit of student ID is **6** → Chapter 6: *Neural Networks*
Source: [Jurafsky & Martin SLP3, Chapter 6](https://web.stanford.edu/~jurafsky/slp3/6.pdf)

### 1.2 Document Processing & Cleaning

The PDF is loaded with **PyMuPDF** (via `langchain_community.document_loaders.PyMuPDFLoader`).
Raw PDF extraction introduces several artefacts that are cleaned before indexing:


| Step                    | What is removed / fixed                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| LaTeX ligatures         | `ﬁ`→`fi`, `ﬀ`→`ff`, `ﬂ`→`fl`, etc.                                                                     |
| Hyphenated line-breaks  | `addi-\ntional` → `additional`                                                                         |
| Page headers            | `2\nCHAPTER 6\n•\nNEURAL NETWORKS\n`                                                                   |
| Standalone page numbers | Bare integers on their own line                                                                        |
| Figure / Table captions | Lines starting with `Figure X.Y` or `Table X.Y`                                                        |
| Equation numbers        | `(6.1)`, `(6.2)` appearing alone on a line                                                             |
| Margin gloss terms      | Isolated 1–3 word lowercase lines (textbook side annotations: `bias term`, `sigmoid`, `activation`, …) |
| Excess whitespace       | Runs of blank lines, multiple spaces, trailing spaces                                                  |


After cleaning the ~261 k-char raw extraction becomes ~64 k chars of clean prose, split into **182 chunks** (500-char windows, 50-char overlap, `RecursiveCharacterTextSplitter`).

### 1.3 QA Pair Generation

20 question-answer pairs are written by hand, covering the main concepts of Chapter 6:


| #     | Topic                                      |
| ----- | ------------------------------------------ |
| 1–2   | Neural unit, bias term                     |
| 3–6   | Activation functions (sigmoid, tanh, ReLU) |
| 7     | XOR problem                                |
| 8     | Feedforward networks                       |
| 9–10  | Backpropagation, gradient descent          |
| 11    | Learning rate                              |
| 12    | Dropout regularisation                     |
| 13–14 | Softmax, cross-entropy loss                |
| 15    | Computation graphs                         |
| 16    | Vanishing gradient problem                 |
| 17    | L2 regularisation                          |
| 18    | Mini-batch SGD                             |
| 19    | Layer roles (input / hidden / output)      |
| 20    | Why non-linearity is necessary             |


---

## Task 2 — Technique Comparison: Naive RAG vs Contextual Retrieval

### Models used


| Component               | Model                                                               |
| ----------------------- | ------------------------------------------------------------------- |
| **Retriever**           | `sentence-transformers/all-MiniLM-L6-v2` (384-dim dense embeddings) |
| **Vector store**        | FAISS flat L2 index                                                 |
| **Generator**           | `Qwen/Qwen2.5-1.5B-Instruct` (open-source, runs locally on CPU/MPS) |
| **Contextual enricher** | Rule-based: section header extracted from document structure        |


### 2.1 Naive RAG

Standard retrieval-augmented generation pipeline:

1. **Chunk** — split cleaned document into 500-char windows with 50-char overlap.
2. **Embed** — encode each chunk with `all-MiniLM-L6-v2`.
3. **Index** — store embeddings in a FAISS flat L2 index.
4. **Retrieve** — at query time, embed the question and find the top-3 nearest chunks.
5. **Generate** — pass the retrieved chunks as context to the language model and return its answer.

**Limitation:** The retriever only sees the raw chunk text. If the question phrasing differs from the chunk wording, relevant chunks may be missed.

### 2.2 Contextual Retrieval

Improves retrieval quality by prepending a short context sentence to each chunk *before* embedding:

```
This chunk from Chapter 6: Neural Networks discusses content from Section 6.1: Units.

<original chunk text>
```

The context prefix is derived by detecting section headers in the cleaned document (e.g. `6.1\nUnits`, `6.3\nFeedforward Neural Networks`) using a regex, then assigning each chunk to the nearest preceding section.
The enriched chunk is embedded and stored in a separate FAISS index; retrieval and generation follow the same steps as Naive RAG.

**Benefit:** The section label makes chunks more distinctive and semantically anchored, so the retriever is more likely to surface the right passage even when question phrasing differs from chunk wording.

### 2.3 Evaluation

All 20 QA pairs are passed through both pipelines. Results are saved to `answer/response-st-xxxxxx-chapter-6.json` in the format required by the assignment:

```json
[
  {
    "question": "What is a neural unit and what computation does it perform?",
    "ground_truth_answer": "...",
    "naive_rag_answer": "...",
    "contextual_retrieval_answer": "..."
  },
  ...
]
```

### 2.4 ROUGE Analysis

ROUGE-1, ROUGE-2, and ROUGE-L F1 scores (stemmed) are computed between each generated answer and its ground-truth answer, then averaged over all 20 pairs.


| Method               | ROUGE-1 | ROUGE-2 | ROUGE-L |
| -------------------- | ------- | ------- | ------- |
| Naive RAG            | —       | —       | —       |
| Contextual Retrieval | —       | —       | —       |


*(Fill in after running the evaluation cell in the notebook.)*

**Discussion:**
Contextual Retrieval is expected to outperform Naive RAG on ROUGE scores because the section-level context prefix helps the retriever locate chunks whose *role in the document* matches the question, not just their local wording. Naive RAG can miss relevant chunks when the question uses different vocabulary from the chunk, because the dense embeddings of bare chunks carry no positional or topical context.

---

## Task 3 — Chatbot Web Application

### 3.1 Web Application

A **Streamlit** chat interface (`app/app.py`) built with `st.chat_input` and `st.chat_message`. Runs entirely offline once the embedding model and PDF are available.

### 3.2 Chatbot Functionality

Users can ask any question about Chapter 6 in natural language. The app maintains a conversation history and renders each exchange in a chat bubble layout.

### 3.3 Backend — Contextual Retrieval

Each user query goes through the same Contextual Retrieval pipeline described in Task 2.2:

1. Query is embedded with `all-MiniLM-L6-v2`.
2. Top-3 enriched chunks are retrieved from the FAISS index.
3. Retrieved context is passed to `Qwen/Qwen2.5-1.5B-Instruct` to generate the answer.

Enriched chunks are cached in `models/enriched_chunks.pkl` after the first run to avoid reprocessing.

### 3.4 Deliverable — Source Citation

Every assistant response includes an expandable **"Source chunks used"** panel that shows the exact passage(s) retrieved from the textbook — allowing the user to verify the answer against the source material.

**Run the app:**

```bash
streamlit run A6/app/app.py
```

---

## Setup

```bash
# Install dependencies
uv sync

# Generate the evaluation JSON (runs locally, no internet needed after first setup)
uv run python A6/code/generate_results.py

# Launch the chatbot
streamlit run A6/app/app.py
```

**Offline model loading** — `all-MiniLM-L6-v2` is loaded from the local HuggingFace cache. The environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set automatically in the notebook and scripts to prevent network requests when behind a proxy.

---

## Dependencies


| Package                    | Purpose                          |
| -------------------------- | -------------------------------- |
| `langchain-community`      | PDF loader (`PyMuPDFLoader`)     |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `sentence-transformers`    | Dense text embeddings            |
| `faiss-cpu`                | Vector similarity search         |
| `rouge-score`              | ROUGE evaluation metrics         |
| `streamlit`                | Web chat interface               |
| `transformers`             | Local open-source language model |
| `pymupdf`                  | PDF parsing backend              |
| `scikit-learn`             | TF-IDF retrieval (fallback)      |


