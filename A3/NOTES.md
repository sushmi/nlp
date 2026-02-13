# NOTES

## DatasetNotFoundError - Multi30K 

The Hugging Face Datasets Hub does not currently host the "multi30k" dataset.

What you can do:

1. Alternative datasets:

Use "wmt14" or "wmt16" for English-German translation tasks.

Example:

```python
from datasets import load_dataset
dataset = load_dataset("wmt14", "de-en")
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]
```

2. Manual download:

- Download Multi30k from its official source: https://www.statmt.org/wmt16/multi30k.html
- Manually preprocess and load the data using pandas or custom code.

3. User other Hugging Face dataset:

- Search for available translation datasets: https://huggingface.co/datasets?task_categories=task_categories:translation

Summary:

"multi30k" is not available on Hugging Face Datasets Hub.
Use "wmt14" or "wmt16" for English-German translation, or manually download Multi30k and load it yourself.


## English to German

The WMT14 dataset only provides "de-en" (German-to-English) as a configuration, not "en-de" (English-to-German). 

If you want English-to-German, you must use "de-en" and reverse the source and target in your code.

How to use for English-to-German:

- Load with "de-en":

- For each sample, use:

```python
sample["translation"]["en"] as the source (English)
sample["translation"]["de"] as the target (German)
```

Summary:

- "en-de" is not available.
- Use "de-en" and treat English as source, German as target.
- Adjust your preprocessing accordingly.

## AttributeError: 'DatasetDict' object has no attribute 'random_split'

The Hugging Face DatasetDict does not have a random_split method. Instead, you should use the built-in splits provided by the dataset, or manually split the dataset using the train_test_split method.

For WMT14, the splits are already provided: "train", "validation", and "test".

Usage:
```python
from datasets import load_dataset

dataset = load_dataset("wmt14", "de-en")
train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]
```

## Tokenizer

To get a tokenizer for English and German, use spaCy. After installing spaCy and downloading the language models, you can create tokenizers like this:

You can use these functions to tokenize sentences in your dataset. 

```
import spacy

# Load spaCy models
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

# Define tokenizer functions
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]
```

```bash
uv pip install spacy

uv run python3 -m spacy download en_core_web_sm
```

## WMT14 dataset is very large - it takes long time to train model

The WMT14 dataset is very large (~4.5M sentence pairs). Increase performance can be done by:

- Using a subset of the training data for vocabulary building


VOCAB_MAX_SAMPLES = 100_000 - Only uses 100k samples instead of 4.5M to build vocabulary. This is sufficient for good coverage.


- Adding progress tracking with tqdm - Show progress during tokenization


Performance of different VOCAB_MAX_SAMPLES size:

- 50_000 - Faster (~15-30 sec), slightly smaller vocab

- 100_000 - Good balance (~30-60 sec)

- 200_000 - Better coverage (~1-2 min)

## Can GPU help in tokenization and vocabulary building

No, GPU won't help here. Vocabulary building is a CPU-bound task (tokenization + word counting), not a matrix computation that GPUs excel at.

This can be speed up significantly with multiprocessing.

Key optimization that can done:

| Change                    | Impact                                         |
|---------------------------|------------------------------------------------|
|spacy.pipe(batch_size=1000)|Batch processing is 5-10x faster than one-by-one|
| disable=["tagger", "parser", "ner", "lemmatizer"] | Only loads tokenizer, much faster |
| VOCAB_MAX_SAMPLES = 50_000 | Reduced from 100k, still good coverage |


Note: GPU helps later for:

- Model training (embedding, GRU, attention)
- Inference

But not for tokenization/vocab building which is string processing.


## What is flores+

Flores is translation dataset maintained by Facebook or meta

FLORES+ is an extended version of FLORES-200 with additional sentences and improved quality. 

Key differences from FLORES-200:

|Feature	|FLORES-200	|FLORES+    |
|-----------|-----------|-----------|
|Languages	|200	    |200+       |
|Sentences	|~2000	    |More       |
|Quality	|Good	    |Improved annotations|
|Splits	    |dev, devtest	|dev, devtest.    |

Note: Like FLORES-200, FLORES+ is primarily for evaluation, not training. For training a translation model, use datasets like:

- wmt14, wmt16, wmt19
- opus100
- tatoeba

## What is WTM

WMT (Workshop on Machine Translation)

```python

from datasets import load_dataset

# WMT14 - Classic benchmark
wmt14 = load_dataset("wmt14", "de-en")  # ~4.5M pairs

# WMT16 - More language pairs
wmt16 = load_dataset("wmt16", "de-en")

# WMT19 - Largest, most recent
wmt19 = load_dataset("wmt19", "de-en")  # ~38M pairs
```

- Source: News articles, EU parliament, web crawl
- Quality: High quality, professionally curated
- Best for: Serious MT research, benchmarking
- Splits: train, validation, test

## OPUS

```python
from datasets import load_dataset

opus = load_dataset("opus100", "en-de")
```

- Source: Aggregated from many sources (subtitles, web, religious texts)
- Quality: Mixed (some noisy)
- Best for: Multilingual models, low-resource languages
- Languages: 100 language pairs with English
- Size: ~1M pairs per language pair

## Tatoeba
- Source: User-contributed sentences (community project)
- Quality: Simple sentences, good for beginners
- Best for: Evaluation, small experiments, rare languages
- Size: Varies by language (1K - 500K pairs)
- Note: Sentences are short and simple

## Comparison table on language translation dataset

|Your Goal	|Best Dataset|
|-----------|------------|
|Learn/Experiment	|Tatoeba (small, simple)|
|Train production model	|WMT19 (large, high quality)|
|Multilingual model	|OPUS-100 (100 languages)|
|Benchmark/Compare	|WMT14 (standard benchmark)|
|Low-resource language	|OPUS-100 or Tatoeba|


## What is SentencePeice

SentencePiece is a tokenization tool or algorithm that trains on data. It's not pretrained model.

```python
import sentencepiece as spm

# Step 1: TRAIN on your data (creates vocabulary)
spm.SentencePieceTrainer.train(
    input='your_corpus.txt',      # Your text file
    model_prefix='my_tokenizer',  # Output name
    vocab_size=8000,              # How many tokens
    model_type='bpe'              # Algorithm type
)

# Step 2: USE the trained tokenizer
sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')
tokens = sp.encode("Hello world", out_type=str)

```

## BPE vs Unigram: Subword Tokenization Algorithms

Both are algorithms to break workds into smaler subword units

BPE (Byte Pair Encoding)

How it works:
1. Start with individual characters
2. Iteratively merge the most frequent pair
3. Repeat until vocabulary size reached

```python
Corpus: "low lower lowest"

Step 0: Characters
['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'l', 'o', 'w', 'e', 's', 't']

Step 1: Most frequent pair = ('l', 'o') → merge to 'lo'
['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'lo', 'w', 'e', 's', 't']

Step 2: Most frequent pair = ('lo', 'w') → merge to 'low'
['low', ' ', 'low', 'e', 'r', ' ', 'low', 'e', 's', 't']

Step 3: Most frequent pair = ('low', 'e') → merge to 'lowe'
['low', ' ', 'lowe', 'r', ' ', 'lowe', 's', 't']

... continue until vocab_size reached

Final vocabulary: ['low', 'er', 'est', 'e', 's', 't', ...]
```

## Unigram

How it works:

1. Start with a large vocabulary (all substrings)
2. Remove tokens that least impact the likelihood
3. Repeat until vocabulary size reached

Example:
```python
Start with large vocab:
['l', 'o', 'w', 'lo', 'low', 'ow', 'lower', 'lowest', 'e', 'r', 'er', 'est', ...]

Calculate probability of each token appearing
Remove least useful tokens one by one

Final vocabulary: ['low', 'er', 'est', ...]
```

Vocabulary Comparison
```python

BPE (Bottom-Up):
────────────────
Characters → Merge frequent pairs → Vocabulary

  l o w e r        Most common: 'lo' → merge
  └─┬─┘            
   lo  w e r       Most common: 'low' → merge
   └──┬──┘         
    low  e r       Most common: 'er' → merge
         └─┬─┘     
    low   er       Final tokens!


Unigram (Top-Down):
───────────────────
All substrings → Remove least useful → Vocabulary

  [l, o, w, lo, low, ow, e, r, er, ...]   Start big
           ↓ Remove 'ow' (rare)
  [l, o, w, lo, low, e, r, er, ...]
           ↓ Remove 'l' (can use 'lo')
  [o, w, lo, low, e, r, er, ...]
           ↓ Keep removing...
  [low, er, ...]                          Final tokens!

```

Key Differences

| Aspect	| BPE	| Unigram |
|-----------|-------|---------|
Approach	|Bottom-up (merge)	|Top-down (remove)
Deterministic?	|✅ Yes, same output	|❌ Probabilistic
Segmentation	|Single best split	|Multiple possible splits
Training	|Faster	|Slower
Used by	|GPT, RoBERTa	|T5, ALBERT, XLM




