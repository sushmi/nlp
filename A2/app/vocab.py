UNKNOWN_TOKEN = "<unk>"
END_OF_SENTENCE_TOKEN = "<eos>"

class Vocab:
    def __init__(self, counter, min_freq=1, specials=None):
        self.itos = []  # index to string
        self.stoi = {}  # string to index
        self.default_index = 0
        
        # Add special tokens first
        if specials:
            for token in specials:
                self._add_token(token)
        
        # Add tokens that meet min_freq threshold
        for token, count in counter.most_common():
            if count >= min_freq:
                if token not in self.stoi:
                    self._add_token(token)
    
    def _add_token(self, token):
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
    
    def set_default_index(self, index):
        self.default_index = index
    
    def get_itos(self):
        return self.itos
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)
    
    def __len__(self):
        return len(self.itos)
