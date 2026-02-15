# NOTES

#### Reading Resources:

1. [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805)
2. [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006.pdf)
3. [The Illustrated BERT, ELMo, and co.](http://jalammar.github.io/illustrated-bert/)
4. [Jurafsky and Martin Chapter 11 (Fine-Tuning and Masked Language Models)](https://web.stanford.edu/~jurafsky/slpdraft/11.pdf)
5. [Slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture09-pretraining-updated.pdf)

Extracted from: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/ 

## What is BERT?

[BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805) 

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed for natural language understanding. Its main innovation is that it reads text bidirectionally, meaning it considers the context from both the left and right of each word, unlike previous models that read text only left-to-right or right-to-left.

Summary of the BERT whitepaper:

- BERT is pre-trained on large text corpora (like Wikipedia and BookCorpus) using two unsupervised tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
- In MLM, some words in a sentence are randomly masked, and the model learns to predict them using the surrounding context.
- In NSP, the model learns to predict if one sentence follows another, helping it understand relationships between sentences.
- After pre-training, BERT can be fine-tuned for various NLP tasks (like question answering, sentiment analysis, and natural language inference) by adding a simple output layer.
- BERT achieved state-of-the-art results on many NLP benchmarks, showing the power of deep bidirectional context.

In short, BERT is used for a wide range of NLP tasks where understanding the meaning and relationships in text is important, thanks to its deep, bidirectional, context-aware representations.


## Feature based approach

Extraction of features (representation or embeddings) from the data. These features are then used as input to anothe rmachine learning model like logistic regression, SVM or a neural network to perform specific task. 

** Model's parameters are not updated during this process

## Fine tuning task

This is the approach where a pre-trained model and add a small, task-specifi layer on top. Then, train (fine-tune) the entire model - including the pre-trained part - on a specific dataset. This allows the model to adapt its learned representations to the new task, often improving performance.

Feature-based: Use pre-trained models as fixed feature extractors.

Fine-tuning: Update the whole model for your specific task.


## What is difference between Fine tuning and hyperparameter tuning ?

Suppose a pre-trained model called pretrained_model and need to classify text.

Fine-tuning:

The model’s weights on new specific data.

```python
# ...existing code...
import torch.nn as nn

# Add a new classification layer on top
model = nn.Sequential(
    pretrained_model,         # Pre-trained model
    nn.Linear(768, 2)         # New layer for binary classification
)

# Fine-tuning: update all parameters
for param in model.parameters():
    param.requires_grad = True

# Training loop (simplified)
for data, label in train_loader:
    output = model(data)
    loss = loss_fn(output, label)
    loss.backward()
    optimizer.step()
# ...existing code...
```

Hyperparameter tuning:

Experiment with settings like learning rate or batch size, but the code for training/fine-tuning stays the same.

```python
# ...existing code...
# Try different learning rates
for lr in [0.1, 0.01, 0.001]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Train the model as above and see which lr works best
# ...existing code...
```

## Does fine tuning mean we pass model through more epochs?

Tuning (updating) model parameters means adjusting the internal weights of the model during training so it learns from the data. This is done by passing domain specific data through the model multiple times (epochs), calculating the loss, and using optimization algorithms (like SGD or Adam) to update the weights to reduce the loss.

So, when training for more epochs, the model sees the data more times and its parameters are updated further, which can help it learn better—up to a point. However, tuning parameters is not just about more epochs; it’s about the whole training process where the model’s weights are changed to fit your specific task.


```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example: a simple neural network
model = nn.Linear(10, 2)  # 10 input features, 2 output classes

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Dummy data: 100 samples, 10 features each
data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))

# Training for multiple epochs
for epoch in range(5):  # 5 epochs
    optimizer.zero_grad()
    outputs = model(data)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()  # <-- This updates the model's parameters
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

```


Fine-tuning on a pre-trained model means you start with a model that has already learned general patterns from a large dataset (like BERT trained on Wikipedia). You then add a small task-specific layer (for example, a classifier for sentiment analysis) and train the entire model—including the pre-trained part—on your own dataset.

During fine-tuning:

The model’s weights (including the pre-trained layers) are updated using new data.
This helps the model adapt its general knowledge to perform well on the specific task.

```python
import torch.nn as nn

# Assume pretrained_model is already loaded (e.g., BERT, ResNet, etc.)
model = nn.Sequential(
    pretrained_model,      # Pre-trained model
    nn.Linear(768, 2)      # New layer for your task (e.g., binary classification)
)

# Enable training for all parameters (including pre-trained)
for param in model.parameters():
    param.requires_grad = True

# Training loop (simplified)
for epoch in range(num_epochs):
    for data, label in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()  # Updates all model weights, including pre-trained ones
```

For example, to fit your domain-related information (like customer reviews for a product) to a pre-trained model, fine-tuning can be used. Here how it can be done:

1. Start with a pre-trained model:
For example, use a BERT model that’s already learned general language patterns.

2. Add a task-specific layer:
Add a new layer (like a Linear layer for classification) on top of the pre-trained model.

3. Prepare domain data:
Collect and label customer reviews (e.g., positive/negative, or categories relevant to the product).

4. Fine-tune the model:
Train the entire model (including the pre-trained part) on the labeled customer reviews. This updates the model’s weights so it learns patterns specific to the domain.

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load pre-trained BERT and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example customer review
review = "This product is amazing!"
inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True)

# Forward pass (for training, use batches and labels)
outputs = model(**inputs)

# During training, use labeled customer reviews and update model weights
# optimizer = ...
# loss = ...
# loss.backward()
# optimizer.step()
```

Summary:
Fine-tuning lets the model adapt its general language knowledge to the specific domain (customer reviews), so it performs better on the task.

 ** Something to think about ? the new layer added on top of the pre-trained model is typically a supervised model. It is trained using labeled data for your specific task (such as sentiment labels for customer reviews). This layer learns to map the features from the pre-trained model to your desired output classes, making the whole system supervised during fine-tuning.


The combination of BERT and a supervised new layer helps because:

- BERT already understands a lot about language from reading huge amounts of text. It knows grammar, word meanings, and context.
- The new supervised layer learns to use BERT’s language knowledge to solve your specific problem (like classifying reviews as positive or negative) using your labeled data.
- When you train (fine-tune) them together, BERT’s general knowledge is adjusted just enough to work really well for your task, while the new layer learns exactly how to make the right predictions.

In simple terms:
BERT gives the model “language smarts,” and the supervised layer teaches it to use those smarts for your job. This makes your model accurate and effective, even if you don’t have a huge amount of labeled data.

## ELMo (Peters et al. 2018a)

ELMo (Embeddings from Language Models) is a deep contextualized word representation model introduced by Peters et al. in 2018. Unlike traditional word embeddings (like Word2Vec or GloVe), which give each word a single fixed vector, ELMo generates word representations that depend on the entire context in which the word appears.

Key points:

- ELMo uses a deep, bidirectional LSTM trained on a large text corpus with a language modeling objective.
- The embedding for each word is computed based on the whole sentence, so the same word can have different vectors in different contexts.
- ELMo embeddings can be used as features for various NLP tasks, improving performance by providing richer, context-aware word representations.

In summary:

ELMo provides dynamic, context-sensitive word embeddings, making it a powerful tool for many NLP applications.


## spaCy

spaCy is an open-source library for advanced Natural Language Processing (NLP) in Python. It provides fast and efficient tools for tasks like tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and text classification. spaCy is widely used for building real-world NLP applications because it is easy to use, supports multiple languages, and is optimized for performance.


### en_core_web_sm 
en_core_web_sm is a small English language model provided by spaCy. It includes pre-trained pipelines for tokenization, part-of-speech tagging, named entity recognition, and other NLP tasks. You load it in spaCy with spacy.load("en_core_web_sm") to process English text efficiently. It is lightweight and suitable for many standard NLP applications.

In spaCy, a pipeline is a sequence of processing steps applied to text. Each step (component) performs a specific NLP task, such as tokenization, part-of-speech tagging, or named entity recognition. The pipeline processes text in order, so you get structured linguistic information from raw text.

To disable other feature and use only tokenization

```python
import spacy
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer])
doc = nlp("This is my text")
tokens = [token.text for token in doc]
```

### Lemmatization
Lemmatization is the process of reducing words to their base or dictionary form (lemma). For example, "running", "ran", and "runs" are all reduced to "run". Lemmatization helps standardize words for analysis, making it easier to compare and process text in NLP tasks.

### To improve performance of spaCy tokenization

1.  Disable unnecessary pipeline components:

 Use disable=["tagger", "parser", "ner", "lemmatizer"] and add nlp.add_pipe("sentencizer") if you only need sentence splitting and tokenization.

2. Use nlp.pipe for batching:

spaCy’s nlp.pipe processes texts in batches and is much faster than looping with nlp(text).

```python
import spacy

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
nlp.add_pipe("sentencizer")

texts = [t for t in dataset_train["text"] if t.strip()]
sentences = []
for doc in nlp.pipe(texts, batch_size=1000):
    sentences.append([sent.text for sent in doc.sents])
```

3. Consider using a faster tokenizer:
For pure tokenization, libraries like Hugging Face’s tokenizers or NLTK’s word_tokenize are much faster than spaCy.


## spaCy vs NLTK

NLTK for tokenization is basic NLP tasks, especially it is lightweight, easy to use, and with minimal dependencies. NLTK is great for prototyping, education, and simple pipelines.

However, spaCy remains popular because:

- It is much faster for large-scale processing (written in Cython).
- It provides robust, production-ready pipelines for tokenization, POS tagging, dependency parsing, NER, and more.
- It has better support for modern NLP workflows, including integration with deep learning libraries.
- Its API is more consistent and user-friendly for large projects.
- It supports efficient batch processing and is designed for real-world, production use.

In summary:

- Use NLTK for lightweight, educational, or simple tasks.
- Use spaCy for speed, scalability, and advanced NLP pipelines in production.
- Many projects use both: NLTK for some tasks, spaCy for others, depending on needs.

## ISSUES or ERRORs


### ValueError: [E088] Text of length 538294333 exceeds maximum of 1000000. 

ValueError: [E088] Text of length 538294333 exceeds maximum of 1000000. The parser and NER models require roughly 1GB of temporary memory per 100,000 characters in the input. This means long texts may cause memory allocation errors. If you're not using the parser or NER, it's probably safe to increase the `nlp.max_length` limit. The limit is in number of characters, so you can check whether your inputs are too long by checking `len(text)`.

Code:

```python
raw_text = "".join(dataset_train['text'])
import spacy

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
doc = nlp(raw_text) # max limit valiaton for memory allocation errors
```

### ValueError: [E030] Sentence boundaries unset.

 You can add the 'sentencizer' component to the pipeline with: nlp.add_pipe('sentencizer'). Alternatively, add the dependency parser or sentence recognizer, or set sentence boundaries by setting doc[i].is_sent_start.


This error means spaCy’s pipeline does not have a component to detect sentence boundaries. By default, spaCy uses the dependency parser or a sentence recognizer to split text into sentences. 

`nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])`

If you disable these components, spaCy cannot determine where sentences start and end.

To fix this, add the sentencizer component to your pipeline:

```python
import spacy
nlp = spacy.load("en_core_web_sm") # this will execute all pipelines and take longer time depending on dataset size - it can take longer than 10s of minutes or even more)
```
OR add sentencizer

```python
import spacy
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
nlp.add_pipe("sentencizer")
doc = nlp("Your text here")
sentences = list(doc.sents)
```


## Performance Optimization

### Make evaluation step faster

The main bottleneck is calling make_batch() 50 times inside the loop — that's all CPU work (random sampling, tokenizing, masking, padding). Optimize by pre-generating all batches first, then running a tight GPU inference loop with non-blocking transfers.

1. Separated CPU and GPU work — all make_batch() calls happen first in a pre-generation loop, then GPU inference runs in a tight loop without CPU bottlenecks interleaved

2. Non-blocking transfers — .to(device, non_blocking=True) allows async CPU→GPU memory transfers that overlap with computation

3. Faster argmax — replaced .data.max(2)[1] with .argmax(dim=2) which is more efficient when you only need the indices (skips computing the max values)

4. Added timing — reports total inference time and per-batch latency so you can see the GPU speedup

The main bottleneck was make_batch() running on CPU inside the inference loop, forcing the GPU to idle between batches. Now the GPU gets fed batches back-to-back.

### Optimize make_batch()

The biggest bottleneck is the while loop randomly picking two indices and hoping they're consecutive for positive pairs. With ~1.8M sentences, the chance of tokens_b_index == tokens_a_index + 1 by random chance is about 1 in 1.8M — so it spins millions of times to find just 3 positive pairs!

The fix: deliberately construct positive pairs (pick index, use index+1) and negative pairs (pick two non-consecutive indices) instead of randomly sampling and checking.


Changes to optimize:

1. Eliminated the random search bottleneck — The old code randomly picked two indices and checked if they were consecutive. With ~1.8M sentences, the probability of a random hit was ~1/1.8M, so it spun millions of iterations just to find 3 positive pairs. Now positive pairs are constructed directly by picking idx and using idx+1.

2. Fixed the masking probability bug — The original code called random() twice independently (if random() < 0.1 then elif random() < 0.8), which skewed the actual probabilities. Now it calls random() once and uses proper thresholds: < 0.1 (10% random), < 0.9 (80% mask), else keep.

3. Pre-cached special token IDs — word2id['[CLS]'] etc. are looked up once instead of every iteration.

4. Skips over-length sequences — Added a guard so pairs exceeding max_len are skipped instead of producing broken padding.

5. Simplified random token replacement — Uses randint(4, vocab_size - 1) directly instead of the roundtrip through id2word and back.