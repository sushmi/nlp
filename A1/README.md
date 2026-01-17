# A1 : That's what I like 

## Folder Structure

```
A1/
├── README.md
├── app/
│   └── (empty - for application code)
├── code/
│   ├── 01 - Word2Vec (Skipgram) from Scratch.ipynb
│   ├── 02 - Word2Ve (Negative Sampling) from Scratch.ipynb
│   ├── 03 - GloVe from Scratch.ipynb
│   └── 04 - GloVe (Gensim).ipynb
└── resources/
    └── A1_That_s_What_I_LIKE.pdf
```


## Theory Behind

J(θ) - The loss function (objective to minimize):

$$J(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\sum_{\substack{-m \leq j \leq m \\ j \neq 0}}\log P(w_{t+j} | w_t; \theta)$$


Sums the negative log-probability over all center words and their context words
The negative sign means we're minimizing loss (which maximizes the log-probability)
T
T = total number of words in corpus
m
m = window size (1 in your case: one word left + one word right)

P(o|c) - The probability of predicting context word 
o
o given center word 
c:

$$P(o|c)=\frac{\exp(\mathbf{u_o^{\top}v_c})}{\sum_{w=1}^V\exp(\mathbf{u_w^{\top}v_c})}$$

This is softmax:

Numerator: exp(u 
o
⊤
​
 v 
c
​ ) ​- dot product similarity between context and center embeddings
Denominator: Sum over all 
V
V vocabulary words - normalizes to get a probability distribution
Higher similarity → higher probability
In your model's forward() method, this is computed as:

scores = numerator for the actual context word
norm_scores = denominator (all vocabulary words)
nll = negative log likelihood = -log(scores / norm_scores)


## The Big Picture
Word2Vec wants to learn word meanings by looking at which words appear together. The idea is: **"You are the company you keep"** — if two words appear near each other often, they probably mean similar things.

### What's a Loss Function?
Think of it like a report card:

- The model makes a guess: "If I see the word 'cat', what's the probability the next word is 'animal'?"
- Loss measures: "How wrong was that guess?"
- Lower loss = better guesses = better learning

### What's Probability P(o|c)?
This is asking: **"Given I know the center word, what's the chance the context word appears?"**

Example:

- Center word: "cat"
- Possible context words: "animal" (likely), "fruit" (unlikely)
- P(animal|cat) = high
- P(fruit|cat) = low

### How They Work Together
1. **Model guesses** P(animal|cat) based on embeddings
2. **Loss calculates** how wrong the guess was
3. **Training updates** embeddings to make P(animal|cat) higher next time
4. **Repeat** thousands of times until the model gets really good

#### In Simple Code Terms
```
# What we want: P(animal|cat) = HIGH
# What we calculate: loss = -log(P(animal|cat))
# What we minimize: Make loss smaller = Make P(animal|cat) bigger
```
The negative log part just converts probability into a number we can minimize. That's it!

## What is an Embedding?

An embedding is just a list of numbers that represents a word.

Instead of storing words as text ("cat", "apple"), we represent them as vectors (lists of numbers).

Simple Example
```
apple = [0.2, 0.8, 0.1]
banana = [0.3, 0.7, 0.2]
orange = [0.25, 0.75, 0.15]
```
Imagine you're learning about fruits:

Each number might represent a hidden property:

- First number: sweetness
- Second number: nutritious
- Third number: firmness

Words with similar meanings have similar numbers.

### In Code

```
self.embedding_v = nn.Embedding(vocab_size, emb_size)
self.embedding_u = nn.Embedding(vocab_size, emb_size)
```

This creates two lookup tables:

- embedding_v: Each word gets a 2D embedding (because emb_size=2)
- embedding_u: Same thing, but different set of numbers

Example for a vocab of 7 words with embedding size 2:

```
word "cat"    → embedding_v = [0.5, 0.9]
word "animal" → embedding_v = [0.52, 0.88]  (similar to cat!)
word "fruit"  → embedding_v = [0.1, 0.2]   (very different)
```

## Why Two Embeddings?

- embedding_v = how the center word is represented
- embedding_u = how the context word is represented

They start random, but training adjusts these numbers so that:

- Words that appear together get similar embeddings
- Words that don't appear together get different embeddings

### Real Use
After training, the embedding becomes the word's meaning. You can:

- Compare embeddings to find similar words
- Use embeddings in other ML models
- Visualize them (like the scatter plot at the end of your notebook)