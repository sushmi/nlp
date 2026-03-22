# A5: Optimization Human Preference & LLM-as-a-Judge

## Overview

This assignment fine-tunes a pre-trained LLM using **Direct Preference Optimization (DPO)** to reduce hallucinations and improve truthfulness, then evaluates the improvement using an **LLM-as-a-Judge** pipeline on the AlpacaEval benchmark.

---

## Model


|                    | Details                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------- |
| Base model         | `Qwen/Qwen2.5-1.5B-Instruct`                                                              |
| Fine-tuning method | DPO with LoRA (PEFT)                                                                      |
| Training dataset   | `jondurbin/truthy-dpo-v0.1` (200 examples)                                                |
| HuggingFace model  | [sushmee/qwen2.5-1.5b-truthy-dpo](https://huggingface.co/sushmee/qwen2.5-1.5b-truthy-dpo) |


---

## Task 1 — Dataset Preparation

- **Dataset**: `jondurbin/truthy-dpo-v0.1` — 1,016 preference pairs designed to teach truthfulness and avoid hallucinations
- Each example contains: `prompt`, `chosen` (factual answer), `rejected` (hallucinated/wrong answer)
- Used 200 training examples (subset for efficiency)

---

## Task 2 — DPO Training

### Hyperparameters


| Parameter              | Value                           |
| ---------------------- | ------------------------------- |
| Base model             | Qwen/Qwen2.5-1.5B-Instruct      |
| Epochs                 | 1                               |
| Batch size             | 1 (grad accum: 4, effective: 4) |
| Learning rate          | 5e-5                            |
| LR scheduler           | Cosine                          |
| Warmup ratio           | 0.1                             |
| Beta (DPO)             | 0.1                             |
| Max length             | 256                             |
| LoRA rank              | 8                               |
| LoRA alpha             | 16                              |
| LoRA target modules    | q_proj, v_proj                  |
| Gradient checkpointing | Yes                             |
| Precision              | float16 (weights)               |


### Training Loss Curve

DPO Training Loss

The training loss decreased steadily over steps, indicating the model successfully learned to assign higher probability to chosen (truthful) responses over rejected (hallucinated) ones.

---

## Task 3 — Model on Hugging Face Hub

The fine-tuned model (LoRA adapters merged into base weights) is available at:

**[https://huggingface.co/sushmee/qwen2.5-1.5b-truthy-dpo](https://huggingface.co/sushmee/qwen2.5-1.5b-truthy-dpo)**

---

## Task 4 — LLM-as-a-Judge Evaluation (AlpacaEval)

### Setup

- **Evaluation dataset**: `tatsu-lab/alpaca_eval` — `helpful_base` subset, 15 samples
- **Judge LLM**: Claude (`claude-haiku-4-5-20251001`) via Anthropic API
- **Judge prompt**: Blind side-by-side comparison — judge outputs `"Model A"`, `"Model B"`, or `"Tie"`

### Results


| Sample ID | Instruction (Truncated) | Winner (Judge) |
| --------- | ----------------------- | -------------- |
| 1         |                         |                |
| 2         |                         |                |
| 3         |                         |                |
| 4         |                         |                |
| 5         |                         |                |
| 6         |                         |                |
| 7         |                         |                |
| 8         |                         |                |
| 9         |                         |                |
| 10        |                         |                |
| 11        |                         |                |
| 12        |                         |                |
| 13        |                         |                |
| 14        |                         |                |
| 15        |                         |                |


> Fill in the table above with your actual results after running Task 4.

### Win Rate

$$\text{Win Rate} = \frac{\text{Model B Wins} + 0.5 \times \text{Ties}}{\text{Total Valid Evaluations}} \times 100$$


| Metric              | Count  |
| ------------------- | ------ |
| Model B (DPO) Wins  |        |
| Model A (Base) Wins |        |
| Ties                |        |
| **Win Rate**        | **_%** |


### Discussion

**Did DPO training successfully improve the model?**

The DPO fine-tuned model (Model B) achieved a Win Rate of **__%** on the AlpacaEval `helpful_base` benchmark judged by Claude.

- A win rate **> 50%** indicates DPO training improved the model's helpfulness and accuracy over the base model.
- A win rate **≈ 50%** suggests neutral impact — the models performed similarly.
- A win rate **< 50%** suggests the DPO training may have over-fitted or required more tuning.

**Key observations:**

- Training on only 200 truthfulness-focused preference pairs for 1 epoch is a lightweight intervention; larger-scale DPO runs typically show stronger gains.
- The LoRA-based approach (rank 8, ~7M trainable params out of 1.5B) is memory-efficient and sufficient for demonstrating preference alignment.
- The `beta=0.1` setting keeps the DPO model close to the reference, avoiding reward hacking while still shifting preference.

---

## Environment

- Python 3.11
- PyTorch 2.10.0 (MPS — Apple Silicon)
- transformers 5.1.0
- trl 0.29.0
- peft (LoRA)

