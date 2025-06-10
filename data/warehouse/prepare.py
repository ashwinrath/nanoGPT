"""
Prepare the warehouse dataset for word-level language modeling.
Instead of encoding characters, we map whitespace-separated words to ints.
Saves train.bin, val.bin containing word token ids, and meta.pkl with vocab.
"""
import os
import pickle
import requests
import numpy as np
import re

# Path setup
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/ashwinrath/char-rnn/master/data/warehouse/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

# Load data
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Word-level tokenization
words = re.findall(r'\b\w+\b', data.lower())  # simplistic word tokenizer
vocab = sorted(set(words))
vocab_size = len(vocab)
print(f"vocab size: {vocab_size:,} words")

# Mappings
stoi = { w: i for i, w in enumerate(vocab) }
itos = { i: w for w, i in stoi.items() }
def encode(s): return [stoi[w] for w in re.findall(r'\b\w+\b', s.lower())]
def decode(l): return ' '.join([itos[i] for i in l])

# Train/val split
n = len(words)
train_words = words[:int(n*0.9)]
val_words = words[int(n*0.9):]
train_ids = np.array([stoi[w] for w in train_words], dtype=np.uint16)
val_ids = np.array([stoi[w] for w in val_words], dtype=np.uint16)
print(f"train has {len(train_ids):,} words")
print(f"val has {len(val_ids):,} words")

# Save bin files
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save vocab
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
