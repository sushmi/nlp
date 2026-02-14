# Assignment NLP

## A1: That’s What I LIKE

Topics Related:


PyTorch Word Embeddings Tutorial, Gensim's word2vec, Stanford GloVe, Spacy

[A1 README](A1/README.md)


## A2: Language Model

LSTM Language Model 

[A2 README](A2/README.md)


## A3: Make Your Own Machine Translation Language

[A3 Readme](A3/README.md)


## A4: Code your own BERT

[A4 Readme](A4/README.md)



# PyTorch on Apple Silicon 

On M3 Mac (Apple Silicon), you use MPS (Metal Performance Shaders) instead of CUDA for GPU acceleration.

```python
import torch

# Check if MPS is available
print(torch.backends.mps.is_available())  # Should be True on M3

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move tensors/models to GPU
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
model = model.to(device)
```

```python
# universal python code to select gpu according to your system
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")      # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon GPU
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
```

PyTorch Version: Requires PyTorch 1.12+ (MPS support added)


# Troubleshoot

##  Add Large files onto Git

```
remote: error: File A2/code/class/best-val-lstm_lm.pt is 294.44 MB; this exceeds GitHub's file size limit of 100.00 MB
```

GitHub usually rejects large files (over 100 MB) or files that are in your .gitignore. model files are generally larger than 100 MB .

If you tracked with Git LFS after the file was already committed (or staged), Git is still trying to push the large file as a normal object, which GitHub rejects.

How to fix:
0. Unstage and remove the large file from Git history:

```
git rm --cached path/to/best-val-lstm_lm.pt
```
1. If you have already committed to remote , you need to use lfs migrate

1.1 Install if you don't have lfs

```
brew install git-lfs
```
1.2 Add to git

```
git lfs install
```

1.3 Use lfs migrate

1.3.1 Track the file type with LFS

```
git lfs track "*.pt"
git add .gitattributes
```
1.3.2 Migrate all existing .pt files in your repo history to LFS

```
git lfs migrate import --include="*.pt"
```

This rewrites your repo history so all .pt files are stored in LFS, not in normal Git.

1.3.3 Force-push the rewritten history to GitHub

```
git push --force origin main
```

1.3.4. Clean up old local backups (optional)

```
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

1.3.5. Tell collaborators to re-clone the repo

Because history changed, others should:

- Backup their work
- Delete their local repo
- Clone fresh from GitHub
