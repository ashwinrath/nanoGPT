"""
Sample from a trained model (word-level)
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out'  # set via CLI
start = "The worker picks"
num_samples = 1
max_new_tokens = 50
temperature = 0.8
top_k = 200
seed = 1337
device = 'cpu'
dtype = 'float32'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval().to(device)
if compile:
    model = torch.compile(model)

# Load word-level encoder/decoder
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']
encode = lambda s: [stoi[w] for w in s.strip().split()]
decode = lambda l: ' '.join([itos[i] for i in l])

# Encode prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# Generate and decode
with torch.no_grad():
    with ctx:
        for _ in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
