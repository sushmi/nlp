# Classification Metrics Explained: A Complete Guide

## Table of Contents
1. [The Basics: What is Classification?](#basics)
2. [Understanding the Confusion Matrix](#confusion-matrix)
3. [Support](#support)
4. [Precision](#precision)
5. [Recall](#recall)
6. [F1-Score](#f1-score)
7. [Accuracy](#accuracy)
8. [MLM vs NSP Metrics](#mlm-vs-nsp)
9. [Practical Examples](#examples)

---

## 1. The Basics: What is Classification? {#basics}

Classification is predicting which category (class) something belongs to:
- **Binary Classification**: 2 classes (e.g., spam/not spam, next/not next)
- **Multi-class Classification**: 3+ classes (e.g., entailment/neutral/contradiction)

**In your BERT model:**
- **NSP (Next Sentence Prediction)**: Binary classification (0=Not Next, 1=Is Next)
- **MLM (Masked Language Modeling)**: Multi-class classification (predict 1 word from ~520,000 words)

---

## 2. Understanding the Confusion Matrix {#confusion-matrix}

The confusion matrix shows **what your model predicted vs. what was actually true**.

### For Binary Classification (NSP):

```
                    PREDICTED
                 Not Next  Is Next
              ┌──────────┬────────┐
    ACTUAL    │          │        │
    Not Next  │    TN    │   FP   │  ← Actual "Not Next" sentences
              ├──────────┼────────┤
    Is Next   │    FN    │   TP   │  ← Actual "Is Next" sentences
              └──────────┴────────┘
```

**Key Terms:**
- **TP (True Positive)**: Model predicted "Is Next" ✓ and it WAS next ✓
- **TN (True Negative)**: Model predicted "Not Next" ✓ and it WASN'T next ✓
- **FP (False Positive)**: Model predicted "Is Next" ✗ but it WASN'T next
- **FN (False Negative)**: Model predicted "Not Next" ✗ but it WAS next

### Example from Your Model:

```
                    PREDICTED
                 Not Next  Is Next
              ┌──────────┬────────┐
    ACTUAL    │          │        │
    Not Next  │    75    │   75   │  ← 150 actual "Not Next"
              ├──────────┼────────┤
    Is Next   │    75    │   75   │  ← 150 actual "Is Next"
              └──────────┴────────┘
```

This shows **random guessing** - the model is 50% correct for each class.

---

## 3. Support {#support}

**Definition**: The number of actual occurrences of each class in the dataset.

**Formula**:
```
Support = Total number of samples in that class
```

**Example from your NSP task:**
```
Support for "Not Next" = 150  (there were 150 actual "Not Next" sentences)
Support for "Is Next"  = 150  (there were 150 actual "Is Next" sentences)
```

**Why it matters:**
- Shows if your dataset is **balanced** (equal support) or **imbalanced**
- If support differs greatly, accuracy can be misleading
- Example: 95 cats, 5 dogs → if you predict "cat" for everything, accuracy = 95%!

---

## 4. Precision {#precision}

**Question it answers:**
> "Of all the samples I predicted as positive, how many were actually positive?"

**Formula:**
```
Precision = TP / (TP + FP)
          = True Positives / All Predicted Positives
```

**Intuitive Meaning:**
- **How trustworthy are my positive predictions?**
- **When I say "Yes", how often am I right?**

### Real-World Example: Email Spam Filter

```
Your filter marked 100 emails as SPAM
├─ 80 were actually spam (TP) ✓
└─ 20 were important emails (FP) ✗

Precision = 80 / 100 = 0.80 = 80%
```

**Low precision = Many false alarms** (you mark good emails as spam!)

### For NSP "Is Next" class:

```
Model predicted 150 sentences as "Is Next"
├─ 75 were actually next (TP) ✓
└─ 75 were NOT actually next (FP) ✗

Precision = 75 / 150 = 0.50 = 50%
```

**Your model's precision is 50% = random guessing**

---

## 5. Recall {#recall}

**Question it answers:**
> "Of all the actual positive samples, how many did I correctly identify?"

**Formula:**
```
Recall = TP / (TP + FN)
       = True Positives / All Actual Positives
```

**Intuitive Meaning:**
- **How many of the real positives did I catch?**
- **Completeness of detection**

### Real-World Example: Cancer Screening Test

```
100 patients actually have cancer
├─ 90 were detected by the test (TP) ✓
└─ 10 were missed by the test (FN) ✗

Recall = 90 / 100 = 0.90 = 90%
```

**Low recall = Missing many cases** (dangerous for cancer!)

### For NSP "Is Next" class:

```
150 sentences were actually "Is Next"
├─ 75 were correctly identified (TP) ✓
└─ 75 were missed (FN) ✗

Recall = 75 / 150 = 0.50 = 50%
```

**Your model's recall is 50% = random guessing**

---

## 6. Precision vs. Recall: The Trade-off

There's usually a **trade-off** between precision and recall:

### Scenario 1: High Precision, Low Recall (Conservative)
```
Cancer Detector (very strict):
- Only flags cases it's 99% sure about
- Precision = 99% (when it says cancer, it's right)
- Recall = 30% (but it misses 70% of cancer cases!)

Use when: False positives are expensive
Example: Spam filter (don't want to mark important emails as spam)
```

### Scenario 2: Low Precision, High Recall (Liberal)
```
Cancer Detector (very cautious):
- Flags anything suspicious
- Precision = 40% (many false alarms)
- Recall = 98% (catches almost all cancer cases)

Use when: False negatives are expensive
Example: Cancer screening (better safe than sorry)
```

### Visual Representation:

```
          🎯 Perfect Model (100% Precision, 100% Recall)
              │
              │
    Precision │     ╱
        ↑     │   ╱
              │ ╱
              ╱──────────→ Recall
            ╱
          ╱ Trade-off curve
        ╱   (can't have both perfect)
```

---

## 7. F1-Score {#f1-score}

**Question it answers:**
> "What's the balance between precision and recall?"

**Formula:**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
         = Harmonic mean of precision and recall
```

**Why harmonic mean?**
- Regular average would give: (90% + 10%) / 2 = 50%
- Harmonic mean gives: 2 × (90 × 10) / (90 + 10) = 18%
- **Harmonic mean punishes extreme imbalances**

### Examples:

```
Case 1: Balanced Model
Precision = 80%
Recall = 80%
F1 = 2 × (80 × 80) / (80 + 80) = 80%  ✓ Good!

Case 2: Imbalanced Model
Precision = 90%
Recall = 10%
F1 = 2 × (90 × 10) / (90 + 10) = 18%  ✗ Poor!

Case 3: Your NSP Model
Precision = 50%
Recall = 50%
F1 = 2 × (50 × 50) / (50 + 50) = 50%  ✗ Random guessing!
```

**When to use F1-Score:**
- You care about both precision AND recall
- You want a single metric to optimize
- Classes are imbalanced

---

## 8. Accuracy {#accuracy}

**Question it answers:**
> "What percentage of all predictions were correct?"

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Correct Predictions / Total Predictions
```

### For NSP:
```
Correct predictions = 75 + 75 = 150
Total predictions = 300
Accuracy = 150 / 300 = 50%
```

### ⚠️ Warning: Accuracy can be misleading!

**Example with imbalanced data:**
```
Dataset: 950 healthy patients, 50 with disease

"Dumb" model: Always predict "healthy"
├─ Correct: 950 (TN)
├─ Wrong: 50 (FN)
└─ Accuracy = 950/1000 = 95%  ← Looks good!

But:
├─ Recall for disease = 0% (missed all disease cases!)
└─ This model is useless for detecting disease!
```

**Lesson**: Use accuracy only when classes are balanced.

---

## 9. MLM vs NSP: Different Metrics for Different Tasks {#mlm-vs-nsp}

### NSP (Next Sentence Prediction) - Binary Classification

**Task**: Predict if sentence B follows sentence A

```
Classes: 2 (Not Next, Is Next)
Samples: 300
```

**Metrics Used:**
- ✅ **Accuracy**: Overall correctness (50%)
- ✅ **Precision**: When predicting "Is Next", how often correct? (50%)
- ✅ **Recall**: Of all actual "Is Next", how many caught? (50%)
- ✅ **F1-Score**: Balance of precision and recall (50%)
- ✅ **Confusion Matrix**: Shows TP, TN, FP, FN
- ✅ **MCC**: Matthews correlation (-1 to +1, handles imbalance)

**Why these metrics?**
- Binary classification benefits from precision/recall/F1
- We care about both classes equally
- Can analyze false positives vs false negatives

### MLM (Masked Language Modeling) - Multi-class Classification

**Task**: Predict the masked word from vocabulary

```
Classes: ~520,000 words
Samples: 1,406 masked tokens
```

**Metrics Used:**
- ✅ **Accuracy**: Percentage of correctly predicted words (4.34%)
- ✅ **Perplexity**: How "surprised" the model is by actual words
- ❌ **Precision/Recall/F1 for each word?** → Impractical with 520k classes!

**Why just accuracy and perplexity?**
- 520,000 classes = confusion matrix would be 520k × 520k!
- Most words appear rarely (support ≈ 0 for most classes)
- Precision/recall for each word is not meaningful
- **Perplexity** is better for language modeling:
  - Measures model confidence
  - Lower = better predictions
  - Your perplexity ≈ 1.3 million = very poor!

### Visual Comparison:

```
┌─────────────────────────────────────────────────────────┐
│ NSP: Binary Classification (2 classes)                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Not Next [████████] 50%   Is Next [████████] 50%      │
│                                                          │
│  ✓ Easy to analyze with precision/recall/F1             │
│  ✓ Confusion matrix is 2×2 (readable)                   │
│  ✓ Can understand model behavior per class              │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ MLM: Multi-class Classification (~520,000 classes)      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  the [█] cat [█] dog [█] ... (519,997 more words)       │
│                                                          │
│  ✗ Too many classes for per-class analysis              │
│  ✗ Confusion matrix is 520k×520k (unreadable)           │
│  ✓ Use accuracy and perplexity instead                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Practical Examples {#examples}

### Example 1: Perfect NSP Model

```
Confusion Matrix:
                 Predicted
              Not Next  Is Next
    Actual
    Not Next     150       0      ← Perfect!
    Is Next        0     150      ← Perfect!

Metrics:
- Accuracy = (150+150)/300 = 100%
- Precision = 150/(0+150) = 100%
- Recall = 150/(0+150) = 100%
- F1-Score = 100%
```

### Example 2: Your Current NSP Model (Random)

```
Confusion Matrix:
                 Predicted
              Not Next  Is Next
    Actual
    Not Next      75      75     ← 50% correct
    Is Next       75      75     ← 50% correct

Metrics:
- Accuracy = (75+75)/300 = 50%  ← Like flipping a coin
- Precision = 75/(75+75) = 50%
- Recall = 75/(75+75) = 50%
- F1-Score = 50%
```

### Example 3: Biased Model (Always predicts "Is Next")

```
Confusion Matrix:
                 Predicted
              Not Next  Is Next
    Actual
    Not Next       0     150     ← All wrong!
    Is Next        0     150     ← All correct!

Metrics:
- Accuracy = 150/300 = 50%      ← Same as random!
- Precision = 150/(150+150) = 50%
- Recall = 150/(0+150) = 100%   ← Caught all "Is Next"!
- F1-Score = 67%                ← Better than random

But this model is useless! It just says "Yes" to everything.
```

---

## Summary Table

| Metric | What It Measures | Formula | Your NSP | Good Value |
|--------|-----------------|---------|----------|------------|
| **Support** | How many samples in class | Count | 150 each | Balanced |
| **Precision** | "When I say yes, am I right?" | TP/(TP+FP) | 50% | >80% |
| **Recall** | "Did I catch all the yes cases?" | TP/(TP+FN) | 50% | >80% |
| **F1-Score** | Balance of precision & recall | 2×P×R/(P+R) | 50% | >80% |
| **Accuracy** | Overall correctness | (TP+TN)/Total | 50% | >80% |
| **Perplexity** | Model uncertainty (MLM) | exp(loss) | 1.3M | <50 |

---

## Key Takeaways

1. **Support** = How many samples exist (not about predictions)
2. **Precision** = Of my positive predictions, how many were right?
3. **Recall** = Of all actual positives, how many did I catch?
4. **F1-Score** = Balance between precision and recall
5. **Accuracy** = Overall percentage correct (but can be misleading!)
6. **NSP uses precision/recall/F1** because it's binary (2 classes)
7. **MLM uses accuracy/perplexity** because it has 520k classes
8. **Your model (50% NSP, 4% MLM)** = needs much more training!

---

## Visualization of Your Results

```
      Perfect Model              Your Model
         100%                      50%
          ┃                         ┃
    ┏━━━━━╋━━━━━┓             ┏━━━━━╋━━━━━┓
    ┃     ┃     ┃             ┃     ┃     ┃
  Prec  Recall  F1          Prec  Recall  F1
   ✓     ✓     ✓             ✗     ✗     ✗
```

Your model needs significantly more training data and epochs to improve!
