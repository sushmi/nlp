"""
A6 – Contextual Retrieval Chatbot (Streamlit)

Task 3:
  1) Simple web app with chat interface
  2) Answers questions about Chapter 6 (Neural Networks)
  3) Backend uses Contextual Retrieval (enriched chunks + FAISS + local LLM)
  4) Displays generated answer AND cites the source chunk(s) used

Run from project root:
    streamlit run A6/app/app.py
"""

import os, re, pickle

# Must be set before torch / tokenizers are imported to prevent segfault
# caused by Rust-based tokenizer parallelism conflicting with Streamlit's fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import streamlit as st
import faiss
import numpy as np
import torch
torch.set_num_threads(1)

from huggingface_hub import snapshot_download
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE       = os.path.dirname(__file__)
PDF_PATH   = os.path.join(BASE, "../data/6.pdf")
CACHE_PATH = os.path.join(BASE, "../models/enriched_chunks.pkl")
TITLE      = "Chapter 6: Neural Networks"
TOP_K      = 3
EMBED_ID   = "sentence-transformers/all-mpnet-base-v2"
GEN_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
SECTION_RE = re.compile(r"^(6\.\d+(?:\.\d+)?)\n([A-Z][^\n]{1,60})$", re.MULTILINE)


# ── Data cleaning ─────────────────────────────────────────────────────────────
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
        if (1 <= len(words) <= 3 and s == s.lower() and s[-1:] not in ".?!:,"
                and not re.search(r"\d", s) and not s.startswith("•")):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def build_section_map(full_text):
    return [(m.start(), m.group(1), m.group(2).strip()) for m in SECTION_RE.finditer(full_text)]

def get_section(chunk, full_text, section_map):
    pos  = full_text.find(chunk[:80])
    prev = [(o, n, t) for o, n, t in section_map if o <= pos] if pos != -1 else []
    return f"Section {prev[-1][1]}: {prev[-1][2]}" if prev else TITLE


# ── Build index (cached across reruns) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading Chapter 6 and building index…")
def build_index():
    # Load & clean PDF
    raw   = PyMuPDFLoader(PDF_PATH).load()
    docs  = [Document(page_content=clean_page(d.page_content), metadata=d.metadata) for d in raw]
    full  = "\n\n".join(d.page_content for d in docs)
    smap  = build_section_map(full)

    # Chunk
    splitter    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    chunk_texts = [c.page_content for c in splitter.split_documents(docs)]

    # Embedding model — load from local snapshot, no network
    embed_path  = snapshot_download(EMBED_ID, local_files_only=True)
    embed_model = SentenceTransformer(embed_path, device="cpu")

    # Contextual enrichment — load from cache or build rule-based
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            enriched = pickle.load(f)
    else:
        enriched = [
            f"This chunk from {TITLE} discusses content from {get_section(c, full, smap)}.\n\n{c}"
            for c in chunk_texts
        ]
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(enriched, f)

    embs  = embed_model.encode(enriched, convert_to_numpy=True, show_progress_bar=False, batch_size=32).astype("float32")
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    # Generator — load from local snapshot, no network
    gen_path  = snapshot_download(GEN_ID, local_files_only=True)
    gen_tok   = AutoTokenizer.from_pretrained(gen_path)
    gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    gen_model.eval()

    return embed_model, index, enriched, gen_tok, gen_model


def answer_question(question, embed_model, index, enriched, gen_tok, gen_model):
    q_emb = embed_model.encode([question], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    _, idxs = index.search(q_emb, TOP_K)
    sources = [enriched[i] for i in idxs[0]]
    context = "\n\n---\n\n".join(sources)

    messages = [
        {"role": "system", "content": "Answer using ONLY the given context. Be concise and accurate."},
        {"role": "user",   "content": f"Context:\n{context[:1500]}\n\nQuestion: {question}"},
    ]
    text   = gen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = gen_tok([text], return_tensors="pt")
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_new_tokens=200,
                                 do_sample=False, pad_token_id=gen_tok.eos_token_id)
    answer = gen_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return answer, sources


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Networks QA", page_icon="🧠", layout="wide")
st.title("🧠 Neural Networks QA — Chapter 6")
st.caption(
    "**Retriever:** `all-mpnet-base-v2`  ·  "
    "**Generator:** `Qwen2.5-1.5B-Instruct`  ·  "
    "**Method:** Contextual Retrieval  ·  "
    "Source: Jurafsky & Martin, SLP3"
)

embed_model, index, enriched, gen_tok, gen_model = build_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander("📄 Source chunks used for this answer"):
                for i, src in enumerate(msg.get("sources", []), 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(src[:600] + ("…" if len(src) > 600 else ""))
                    st.divider()

if prompt := st.chat_input("Ask a question about Chapter 6: Neural Networks…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            answer, sources = answer_question(prompt, embed_model, index, enriched, gen_tok, gen_model)
        st.markdown(answer)
        with st.expander("📄 Source chunks used for this answer"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(src[:600] + ("…" if len(src) > 600 else ""))
                st.divider()

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })
