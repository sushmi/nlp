# A6 — uv / Python (project setup)

Same content as **A2/NOTES.md** → section **uv / Python (project setup)**. Kept here for assignment A6.

Notes for this repo (`assignment-npl`): `pyproject.toml`, `uv.lock`, `.venv`, `.python-version`.

### Pin a package version

```bash
uv add "langchain==1.2.13"           # exact
uv add "langchain>=1.2,<2"           # range (quote if needed)
uv add langchain@1.2.13              # alternative form
```

Edit `pyproject.toml` by hand, then run `uv lock` and `uv sync`.

### Check an installed version

```bash
uv pip show langchain
uv pip list | grep -i langchain
```

### Check Python compatibility (example: LangChain)

```bash
uv run python -c "from importlib.metadata import metadata as m; print(m('langchain').get('Requires-Python'))"
```

PyPI also lists **Requires: Python** on each project page.

### Why `uv sync`

- Makes `.venv` match **`uv.lock`** (and `pyproject.toml`): install / upgrade / remove packages so the env matches the lockfile.
- Use after **clone**, **git pull**, or when someone else changed dependencies. `uv add` / `uv remove` already update lock + env; `uv sync` is the “catch up” command.

### PyPI name typos (common errors)

| Wrong | Correct |
|--------|---------|
| `bitstandbytes` | **`bitsandbytes`** |
| `accelarate` | **`accelerate`** |
| `faiss-gup` | **`faiss-gpu`** (GPU / CUDA) or **`faiss-cpu`** (CPU) |

Do **not** install `faiss-cpu` and `faiss-gpu` together; pick one. On macOS, **`faiss-cpu`** is typical; **`faiss-gpu`** is aimed at **Linux + NVIDIA CUDA**.

Example:

```bash
uv add pymupdf faiss-cpu
```

### Install using wheels only (no building sdists)

Prefer wheels by default; to **enforce** wheel-only:

```bash
uv pip install --only-binary :all: package-name
uv add package-name --no-build
```

Environment variable: `UV_NO_BUILD=1` (don’t build source distributions).

### Where `uv python install` puts interpreters

Not inside the project `.venv`. Managed Pythons go under the uv data dir, e.g.:

```bash
uv python dir    # e.g. ~/.local/share/uv/python on macOS/Linux
```

Override: `UV_PYTHON_INSTALL_DIR` or `uv python install 3.11 --install-dir /path`.

Project env: **`./.venv`** in the repo (created by `uv sync` / `uv venv`).

### Python 3.10 vs 3.11 in this project

- **`numpy>=2.4.1`** (as in this repo) requires **Python ≥3.11** for those versions. If you set `requires-python` to include **3.10**, the resolver may fail until you **lower the NumPy upper bound** (e.g. `numpy>=2.0,<2.4.1`) and re-check other pins.
- **LangChain 1.x** metadata is **`>=3.10,<4`**; **3.9** is below that floor for current 1.x.

### Downgrade / switch to Python 3.10 with uv

1. Install and pin:

   ```bash
   uv python install 3.10
   uv python pin 3.10
   ```

2. Set **`requires-python`** in `pyproject.toml` so it matches what your deps support, e.g. `>=3.10,<3.11` if you only use 3.10.

3. Adjust **NumPy** (and anything else) so 3.10 is satisfiable; then:

   ```bash
   rm -rf .venv
   uv lock
   uv sync
   ```

4. Verify:

   ```bash
   uv run python -V
   ```

Alternative explicit venv: `uv venv --python 3.10` then `uv sync`.

### Install Python into the project venv

You don’t install a second Python *inside* `.venv`; the venv **uses** one interpreter. Recreate the env and install deps:

```bash
uv sync
```

Or after pinning: `rm -rf .venv` then `uv sync`.

### Resolver hints (`python_full_version == '3.14'`, `win32`, …)

`uv` can resolve for **all** Python versions allowed by `requires-python`. Tightening helps, e.g. `requires-python = ">=3.11,<3.14"`, if you hit odd cross-platform resolution noise.

## QA

### what is FAISS ?

FAISS (Facebook AI Similarity Search) is a library developed by Meta for efficient similarity search and clustering of dense vectors.

Core Idea

Given a large collection of vectors, FAISS lets you quickly find the k nearest neighbors to a query vector — much faster than brute-force comparison.

Key Concepts

- Indexing: Vectors are stored in an index structure that trades some accuracy for speed
- Similarity metrics: Supports L2 (Euclidean) distance and inner product (cosine similarity)
- Scalability: Handles billions of vectors efficiently

Common Index Types

|Index	|Speed	|Accuracy	|Memory|
|--------|--------|-----------|------|
|IndexFlatL2| 	Slow (exact)	| 100%	| High|
|IndexIVFFla| Fast (approximate)	| ~95%	|Medium|
|IndexIVFPQ	| Fastest	| ~90%	| Low|

Why It Matters for NLP/RAG

In your A6 notebook (RAG assignment), FAISS is likely used to:

Store embeddings — encode documents as dense vectors
Retrieve chunks — find top-k most similar passages to a query
Enable semantic search — go beyond keyword matching
Quick Example
```python
import faiss
import numpy as np

dim = 768  # embedding size (e.g., BERT)
index = faiss.IndexFlatL2(dim)

# Add document embeddings
index.add(doc_embeddings)  # shape: (N, 768)

# Search
distances, indices = index.search(query_embedding, k=5)
```

It's essentially the backbone of vector databases like Pinecone, Weaviate, and ChromaDB — those are largely FAISS with added infrastructure.


### is faiss is embedding model?

No, FAISS is not an embedding model.

The Difference
||Role	|Example|
|-|-|-|
|Embedding Model	|Converts text → vector	|text-embedding-ada-002, sentence-transformers|
|FAISS	|Stores & searches vectors	|finds nearest vectors fast|
How They Work Together

`Text → [Embedding Model] → Vector → [FAISS Index] → Similarity Search`

- Embedding model does the "understanding" — turns text into numbers
- FAISS does the "searching" — finds similar vectors efficiently

Analogy
-Embedding model = a translator (converts meaning into numbers)
-FAISS = a search engine (finds similar numbers fast)

FAISS has no idea what text means — it only works with vectors that an embedding model already produced.

### why the chunks generate are bogus like ffiUffisffiiffinffigffi ffieffimffibffieffidffidffiiffinffigffisffi ffiaffilffilffioffiwffisffi ffinffieffiuffirffiaffilffi ffilffiaffinffigffiuffiaffigffieffi ffimffioffidffieffilffisffi ffitffioffi ffigffieffinffieffirffiaffilffiiffizffieffi ffibffieffitffitffieffirffi ffitffioffi ffiuffinffisffieffieffinffi

using PyMuPDFLoader and the PDF has a ligature encoding problem.

What's Happening
The PDF uses typographic ligatures (ffi, fi, fl) as single glyphs. PyMuPDF can't decode the font mapping, so it outputs the ligature code ffi before every character:


"Using" → "ffiUffisffiiffinffigffi"
Fix: Switch PDF Loader
Option 1 — Use pypdf (simplest):


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('../data/6.pdf')
document = loader.load()
Option 2 — Use pdfplumber (best for complex PDFs):


from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader('../data/6.pdf')
document = loader.load()
Option 3 — Post-process to strip ligatures (quick patch):


import re

def clean_ligatures(text):
    return re.sub(r'ffi|ffl|fi|fl|ff', lambda m: m.group().replace('ffi','').replace('ffl','').replace('fi','').replace('fl','').replace('ff',''), text)

# Or simpler — just remove the "ffi" artifacts:
def clean_text(text):
    return text.replace('ffi', '').replace('ffl', '').replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')
Recommended
Try Option 1 (PyPDF) first — it handles Stanford NLP PDFs well. If still garbled, try pdfplumber.


### why do we need lgatures and what is it explain

What are Ligatures?
Ligatures are single glyphs that replace two or more letters that look awkward when printed side by side.

Why They Exist (Typography)
Certain letter combinations collide visually when printed:


f + i  →  the dot of 'i' crashes into the top of 'f'
f + l  →  same problem
f + f + i  →  triple collision

So typographers designed merged glyphs:

|Letters	|Ligature	|Why|
|-|-|-|
|f + i	|ﬁ	|f hook overlaps i dot|
|f + l	|ﬂ	|f hook overlaps l ascender|
|f + f + i	|ﬃ	|combined mess|
|f + t	|ﬅ	|old style typography|

The PDF Problem

PDFs store ligatures as single special Unicode characters (e.g., ﬁ = U+FB01). When a PDF is exported with a bad font encoding, extractors like PyMuPDF see:

The ligature character ﬁ but can't map it → outputs the ligature name ffi as raw text
This gets prepended to every character → garbled output

"finding" → f+i = ﬁ ligature → extraction fails → "ffifinffidffiffinffigffi"

In Short
- Ligatures = a print/design feature for clean typography
- They are invisible to readers in well-rendered text
- They become a problem only when software extracts text from PDFs without proper font mapping