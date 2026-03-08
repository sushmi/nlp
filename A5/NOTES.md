# NOTES

## Making MPS Deterministic in PyTorch
MPS (Metal Performance Shaders) on Apple Silicon has limited determinism support compared to CUDA. Here's what you can do:

```python
import torch
import os
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # MPS-specific
    torch.mps.manual_seed(seed)
    
    # These help with determinism (may impact performance)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

```

Key caveats for MPS:

- torch.use_deterministic_algorithms(True) — MPS has partial support only; some ops will raise errors if not implemented deterministically
- Unlike CUDA, there's no torch.backends.mps.deterministic flag
- Some ops (e.g., scatter, certain convolutions) may still be non-deterministic on MPS

Safer approach — use deterministic mode with fallback:

```python
import torch
import os

def set_deterministic(seed: int = 42):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Only enable if your ops support it; wrap in try/except
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Full determinism not available: {e}")

set_deterministic(42)
```

Practical notes:

- Setting the seed before each training run is the most reliable approach
- True bit-for-bit reproducibility on MPS is not always guaranteed, even with seeds set
- If reproducibility is critical, consider running on CPU for debugging


## DPO Trainer

DPO (Direct Preference Optimization) is a fine-tuning method for aligning LLMs with human preferences — a simpler alternative to RLHF (Reinforcement Learning from Human Feedback).

### How it works
Instead of training a separate reward model + PPO loop (RLHF), DPO directly optimizes the language model using preference pairs:

` (prompt, chosen_response, rejected_response)`

It uses a closed-form loss derived from the Bradley-Terry preference model:


`L_DPO = -E[ log σ( β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`

Where:

- π_θ = model being trained
- π_ref = frozen reference model (usually the SFT checkpoint)
- y_w = chosen (preferred) response
- y_l = rejected (less preferred) response
- β = temperature controlling deviation from reference

### DPOTrainer (from TRL library)

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    beta=0.1,               # KL penalty coefficient
    output_dir="./dpo_output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
)

trainer = DPOTrainer(
    model=model,            # model to train
    ref_model=ref_model,    # frozen reference model (or None for implicit ref)
    args=training_args,
    train_dataset=dataset,  # needs: prompt, chosen, rejected columns
    tokenizer=tokenizer,
)

trainer.train()
```

Required dataset format

```python
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "I don't know the capital of France."
}
```
vs RLHF

| |RLHF	|DPO|
Reward model	Required	Not needed
RL training	PPO	No RL, direct loss
Stability	Tricky	More stable
Simplicity	Complex	Simple
DPO is widely used for preference alignment because it's stable, simple, and avoids the complexity of RL training loops.