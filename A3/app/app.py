"""
Flask Web Application for English to Nepali Translation
Using Seq2Seq with Attention Model
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify

# ============== Model Definitions ==============

class GeneralAttention(nn.Module):
    """General (Dot-Product) Attention"""
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.proj = nn.Linear(enc_hid_dim, dec_hid_dim)
        
    def forward(self, hidden, encoder_outputs, mask):
        encoder_projected = self.proj(encoder_outputs)
        encoder_projected = encoder_projected.permute(1, 0, 2)
        hidden = hidden.unsqueeze(2)
        attention = torch.bmm(encoder_projected, hidden).squeeze(2)
        attention = attention.masked_fill(mask, -1e10)
        return torch.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc(torch.cat((embedded, output, weighted), dim=1))
        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2SeqPackedAttention(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src == self.src_pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src, src_len)
        input_ = trg[0, :]
        mask = self.create_mask(src)
        
        for t in range(1, trg_len):
            output, hidden, attention = self.decoder(input_, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention
            top1 = output.argmax(1)
            input_ = top1
            
        return outputs, attentions


# ============== Vocab Class ==============

class Vocab:
    """Simple Vocab class"""
    def __init__(self, stoi, itos, default_index=0):
        self.stoi = stoi
        self.itos = itos
        self.default_index = default_index
    
    def __call__(self, tokens):
        return [self.stoi.get(token, self.default_index) for token in tokens]
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)
    
    def set_default_index(self, index):
        self.default_index = index
    
    def get_itos(self):
        return self.itos


# ============== Flask App ==============

app = Flask(__name__)

# Global variables for model and tokenizers
model = None
vocab_src = None
vocab_trg = None
spacy_model = None
nllb_tokenizer = None
device = None

# Special tokens
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_models():
    """Load the translation model and tokenizers"""
    global model, vocab_src, vocab_trg, spacy_model, nllb_tokenizer, device
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load spaCy for English tokenization
    import spacy
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "en_core_web_sm")
    if os.path.exists(os.path.join(model_path, "config.cfg")):
        spacy_model = spacy.load(model_path, disable=["parser", "tagger", "ner", "lemmatizer"])
    else:
        spacy_model = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"])
    
    # Load NLLB tokenizer for Nepali
    from transformers import AutoTokenizer
    nllb_path = os.path.join(os.path.dirname(__file__), "..", "models", "nllb-tokenizer")
    if os.path.exists(nllb_path):
        nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_path)
    else:
        nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    
    # Load vocabularies (you need to save these from training)
    vocab_path = os.path.join(os.path.dirname(__file__), "..", "models")
    
    try:
        vocab_data = torch.load(os.path.join(vocab_path, "vocabs.pt"), weights_only=False)
        vocab_src = Vocab(vocab_data['en_stoi'], vocab_data['en_itos'])
        vocab_src.set_default_index(UNK_IDX)
        vocab_trg = Vocab(vocab_data['ne_stoi'], vocab_data['ne_itos'])
        vocab_trg.set_default_index(UNK_IDX)
    except FileNotFoundError:
        print("WARNING: Vocabulary file not found. Please run training first and save vocabularies.")
        return False
    
    # Model parameters (must match training)
    input_dim = len(vocab_src)
    output_dim = len(vocab_trg)
    emb_dim = 256
    hid_dim = 512
    dropout = 0.5
    
    # Build model
    attn = GeneralAttention(hid_dim * 2, hid_dim)
    enc = Encoder(input_dim, emb_dim, hid_dim, dropout)
    dec = Decoder(output_dim, emb_dim, hid_dim, dropout, attn)
    model = Seq2SeqPackedAttention(enc, dec, PAD_IDX, device)
    
    # Load trained weights
    model_file = os.path.join(vocab_path, "Seq2SeqPackedAttention.pt")
    try:
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return True
    except FileNotFoundError:
        print(f"WARNING: Model file not found at {model_file}. Please train the model first.")
        return False


def translate_sentence(sentence, max_len=50):
    """Translate an English sentence to Nepali"""
    global model, vocab_src, vocab_trg, spacy_model, device
    
    if model is None:
        return "Model not loaded. Please train the model first.", []
    
    model.eval()
    
    # Tokenize input
    tokens = [tok.text.lower() for tok in spacy_model(sentence)]
    
    # Convert to indices
    src_indices = [SOS_IDX] + vocab_src(tokens) + [EOS_IDX]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indices)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    
    mask = model.create_mask(src_tensor)
    
    trg_indices = [SOS_IDX]
    attentions = []
    
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        
        attentions.append(attention.cpu().numpy())
        
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
    
    # Convert indices to tokens
    trg_tokens = [vocab_trg.get_itos()[idx] for idx in trg_indices[1:-1]]  # Remove SOS and EOS
    
    # Join tokens (handle subword tokens from NLLB)
    translation = "".join(trg_tokens).replace("▁", " ").strip()
    
    return translation, attentions


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    english_text = data.get('text', '')
    
    if not english_text.strip():
        return jsonify({'error': 'Please enter some text to translate', 'translation': ''})
    
    translation, _ = translate_sentence(english_text)
    return jsonify({'translation': translation, 'original': english_text})


if __name__ == '__main__':
    print("Loading models...")
    success = load_models()
    if not success:
        print("\n" + "="*60)
        print("NOTE: Model/vocabulary files not found.")
        print("The app will start but translations won't work until you:")
        print("1. Train the model in the notebook")
        print("2. Save vocabularies using the save script")
        print("="*60 + "\n")
    
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
