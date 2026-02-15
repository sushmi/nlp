# Classification Metrics - Quick Reference Card

## 🎯 The 4 Core Metrics (for NSP)

```
┌─────────────┬──────────────────────────┬─────────────┬──────────────┐
│   Metric    │     What It Asks         │   Formula   │  Your Score  │
├─────────────┼──────────────────────────┼─────────────┼──────────────┤
│ PRECISION   │ When I say "Yes",        │ TP/(TP+FP)  │    50%       │
│             │ am I usually right?      │             │  (random)    │
├─────────────┼──────────────────────────┼─────────────┼──────────────┤
│ RECALL      │ Of all "Yes" cases,      │ TP/(TP+FN)  │    50%       │
│             │ how many did I catch?    │             │  (random)    │
├─────────────┼──────────────────────────┼─────────────┼──────────────┤
│ F1-SCORE    │ What's the balance       │ 2×P×R/(P+R) │    50%       │
│             │ between Prec & Recall?   │             │  (random)    │
├─────────────┼──────────────────────────┼─────────────┼──────────────┤
│ SUPPORT     │ How many samples         │ Count       │  150 each    │
│             │ in this class?           │             │  (balanced)  │
└─────────────┴──────────────────────────┴─────────────┴──────────────┘
```

## 🔢 Confusion Matrix (NSP)

```
                    PREDICTED
                Not Next   Is Next
              ┌──────────┬─────────┐
    ACTUAL    │          │         │
    Not Next  │   TN     │   FP    │  ← False Positive = Said "Yes" but was "No"
              ├──────────┼─────────┤
    Is Next   │   FN     │   TP    │  ← False Negative = Said "No" but was "Yes"
              └──────────┴─────────┘
                   ↑
          False Negative
```

**Your Model:**
```
              Not Next   Is Next
    Not Next     75        75      ← 50% accuracy (like flipping coin)
    Is Next      75        75      ← 50% accuracy (like flipping coin)
```

## 📐 Formulas with Visual Breakdown

### Precision
```
          TP                     Correct "Is Next" predictions
    ─────────────  =  ───────────────────────────────────────────
      TP + FP           All times you predicted "Is Next"

    "When I say 'Is Next', how often am I right?"
```

### Recall
```
          TP                     Correct "Is Next" predictions
    ─────────────  =  ───────────────────────────────────────────
      TP + FN          All actual "Is Next" in the dataset

    "Of all real 'Is Next', how many did I find?"
```

### F1-Score
```
         2 × Precision × Recall
    ───────────────────────────
      Precision + Recall

    "Harmonic mean - punishes imbalance!"
```

## 📊 When to Use Each Metric

| Situation | Best Metric | Why |
|-----------|-------------|-----|
| **Balanced classes** (50/50 split) | Accuracy or F1 | Both reliable |
| **Imbalanced classes** (95/5 split) | F1-Score, Precision, Recall | Accuracy misleading! |
| **False positives costly** (spam filter) | Precision | Don't mark good emails as spam |
| **False negatives costly** (cancer test) | Recall | Don't miss sick patients |
| **Both errors matter equally** | F1-Score | Best overall balance |

## 🎭 NSP vs MLM Metrics

```
┌──────────────────────────────────────────────────────────┐
│  NSP (Next Sentence Prediction)                          │
├──────────────────────────────────────────────────────────┤
│  • 2 classes (Not Next, Is Next)                         │
│  • Use: Accuracy, Precision, Recall, F1, Confusion Mx    │
│  • Your score: 50% (random guessing)                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  MLM (Masked Language Modeling)                          │
├──────────────────────────────────────────────────────────┤
│  • ~520,000 classes (every word)                         │
│  • Use: Accuracy, Perplexity                             │
│  • Your score: 4.34% accuracy, perplexity = 1.3M (bad)  │
└──────────────────────────────────────────────────────────┘
```

## 🚦 Score Interpretation

```
Score        NSP Performance    What It Means
────────────────────────────────────────────────────────
100%         🟢 Perfect         Impossible in practice
90-99%       🟢 Excellent       State-of-the-art models
80-89%       🟢 Good            Well-trained model
70-79%       🟡 Fair            Needs improvement
60-69%       🟡 Poor            Undertrained
50-59%       🔴 Very Poor       Barely better than random
50%          🔴 Random          Your current model! ← Need more training
<50%         🔴 Worse than random  Something is broken
```

## 💡 Real-World Analogies

### Precision
```
🎯 Archery: Of the arrows that hit the target, how many hit the bullseye?
   High Precision = Tight grouping (even if off-center)
```

### Recall
```
🎣 Fishing: Of all the fish in the pond, how many did you catch?
   High Recall = Caught most fish (even if you caught some boots too)
```

### F1-Score
```
⚖️  Balance: Good at both aiming (precision) AND catching (recall)
   High F1 = Tight grouping near bullseye AND caught most fish
```

## 🔧 How to Use 

```python
from sklearn.metrics import classification_report

# For NSP evaluation
print(classification_report(
    y_true=nsp_labels,           # Actual labels
    y_pred=nsp_predictions,      # Model predictions
    target_names=['Not Next', 'Is Next']
))

# Output will show:
#              precision    recall  f1-score   support
#   Not Next       0.50      0.50      0.50       150
#     Is Next       0.50      0.50      0.50       150
```

## 🎓 Summary Cheat Sheet

```
When someone asks:                  Answer with:
─────────────────────────────────────────────────────
"How good is your model?"           → Accuracy or F1-Score
"How trustworthy are positives?"    → Precision
"How many positives did you find?"  → Recall
"What's the balance?"               → F1-Score
"How many samples per class?"       → Support
"Show me the errors"                → Confusion Matrix
```

## 🚀 Improving Your BERT Model

Your current scores (NSP: 50%, MLM: 4%) mean you need:

1. ✅ **More training data** - Use 10k-100k samples, not tiny batches
2. ✅ **More epochs** - Train for 1000+ epochs
3. ✅ **Smaller vocabulary** - Use WordPiece (30k words) not raw words (520k)
4. ✅ **Better tokenization** - Use BertTokenizer from HuggingFace
5. ✅ **Learning rate scheduling** - Add warmup and decay

**Goal Metrics:**
- NSP Accuracy: >85%
- NSP F1-Score: >85%
- MLM Accuracy: >60%
- MLM Perplexity: <20

---

## 📚 Resources in This Folder

1. **classification_metrics_explained.md** - Detailed explanations with examples
2. **metrics_examples.py** - Interactive Python examples
3. **bert_evaluation_metrics.py** - Complete evaluation code for your BERT

**To run examples in notebook:**
```python
%run metrics_examples.py
# Or
from metrics_examples import run_all_examples
run_all_examples()
```

---

*Remember: 50% = coin flip. Your model needs training! 🎲*
