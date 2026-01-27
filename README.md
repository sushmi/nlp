# Assignment NLP

Course: http://www.chaklam.com/nlp 


## A1: That’s What I LIKE

Topics Related:


PyTorch Word Embeddings Tutorial, Gensim's word2vec, Stanford GloVe, Spacy

[A1 README](A1/README.md)


## A2: Language Model


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
1. Unstage and remove the large file from Git history:

```
git rm --cached path/to/best-val-lstm_lm.pt
```

2. Re-add the file (now tracked by LFS):

3. Commit the change:

4. Push again:

