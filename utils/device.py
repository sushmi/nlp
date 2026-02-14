# universal device selection: use gpu if available, else cpu
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")      # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon GPU
    else:
        return torch.device("cpu")
    


if __name__ == "__main__":
    device = get_device()

    print(f"Using device: {device}")