"""
Script to save vocabularies from the trained notebook for use in Flask app
Run this in the notebook after training to save the vocabularies
"""

import torch
import os

def save_vocabularies(vocab_transform, save_dir="../models"):
    """
    Save vocabularies for use in the Flask app
    
    Usage in notebook:
        from save_vocab import save_vocabularies
        save_vocabularies(vocab_transform)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vocab_data = {
        'en_stoi': vocab_transform['en'].stoi,
        'en_itos': vocab_transform['en'].itos,
        'ne_stoi': vocab_transform['ne'].stoi,
        'ne_itos': vocab_transform['ne'].itos,
    }
    
    save_path = os.path.join(save_dir, "vocabs.pt")
    torch.save(vocab_data, save_path)
    print(f"Vocabularies saved to {save_path}")
    print(f"  EN vocab size: {len(vocab_data['en_itos'])}")
    print(f"  NE vocab size: {len(vocab_data['ne_itos'])}")


# Code to add to notebook cell after training:
NOTEBOOK_CODE = '''
# Save vocabularies for Flask app
import torch
import os

save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

vocab_data = {
    'en_stoi': vocab_transform[EN_LANGUAGE].stoi,
    'en_itos': vocab_transform[EN_LANGUAGE].itos,
    'ne_stoi': vocab_transform[NE_LANGUAGE].stoi,
    'ne_itos': vocab_transform[NE_LANGUAGE].itos,
}

torch.save(vocab_data, os.path.join(save_dir, "vocabs.pt"))
print("Vocabularies saved!")

# Model is already saved during training to models/Seq2SeqPackedAttention.pt
'''

if __name__ == "__main__":
    print("Add this code to your notebook after training:")
    print("="*50)
    print(NOTEBOOK_CODE)
