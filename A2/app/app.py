
from flask import Flask, render_template_string, request
import torch
from lstm import LSTMLanguageModel
import pickle
from vocab import Vocab
from tokenizer import basic_english_tokenizer
# Import model and vocab loading code here from checkpoint


# This is my choice; change if you have GPU support
device_type = 'cpu'
device = torch.device('cpu')
model_path = "../model"
vocab_filename = "a2_vocab_lm.pkl"

# Loading the vocabulary from the saved file
with open(f'{model_path}/{vocab_filename}', 'rb') as f:
    loaded_vocab = pickle.load(f)


def load_model():
    # Load checkpoint
    device = torch.device(device_type)
    checkpoint = torch.load(f'{model_path}/lstm_lm_checkpoint.pt', map_location=device)

    # Recreate model with saved hyperparameters
    loaded_model = LSTMLanguageModel(
        checkpoint['vocab_size'],
        checkpoint['emb_dim'],
        checkpoint['hid_dim'],
        checkpoint['num_layers'],
        checkpoint['dropout_rate'],
        device
    ).to(device)

    # Load weights
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    print("Model loaded from checkpoint!")

    return loaded_model

def process_prompt(prompt_str):
    max_seq_len = 30
    seed = 0

    return_text = []
    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for temperature in temperatures:
        generated_text = generate(prompt_str, max_seq_len, temperature, model, vocab, device, seed)
        if generated_text[0] == '<unk>':
             generated_text[0] = "[[UNKNOWN]]"
        formatted = f"<div><b>Temperature {temperature}:</b><br><span style='color:#222'>{' '.join(generated_text)}</span></div>"
        return_text.append(formatted)
        
    print(return_text)
    return '\n'.join(return_text)

def generate(prompt, max_seq_len, temperature, model, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = basic_english_tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

# Load your model and vocab here (update as needed)
model = load_model()  # Replace with actual model loading
vocab = loaded_vocab  # Replace with actual vocab loading

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
	<title>A2 Language Model Demo</title>
	<style>
		body { font-family: Arial, sans-serif; margin: 40px; }
		.container { max-width: 600px; margin: auto; }
		input[type=text] { width: 100%; padding: 10px; font-size: 1.1em; }
		button { padding: 10px 20px; font-size: 1.1em; }
		.output { margin-top: 20px; padding: 15px; background: #f4f4f4; border-radius: 5px; }
	</style>
</head>
<body>
	<div class="container">
		<h2>A2 Language Model Text Generation</h2>
		<form method="post">
			<label for="prompt">Enter your prompt:</label><br>
			<input type="text" id="prompt" name="prompt" value="{{ prompt|default('') }}" required><br><br>
			<button type="submit">Generate</button>
		</form>
		{% if output %}
		<div class="output">
			<strong>Generated Text:</strong><br>
            {{ output|safe }}
		</div>
		{% endif %}
	</div>
</body>
</html>
"""
tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}

@app.route('/', methods=['GET', 'POST'])
def index():
	output = None
	prompt = ''
	if request.method == 'POST':
		prompt = request.form['prompt']
		output = process_prompt(prompt)
	return render_template_string(HTML_TEMPLATE, output=output, prompt=prompt)

if __name__ == '__main__':
	app.run(debug=True)
