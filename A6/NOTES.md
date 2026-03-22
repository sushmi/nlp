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
