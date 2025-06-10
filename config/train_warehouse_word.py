# train a word-level GPT model on warehouse dataset

out_dir = 'out-warehouse-word'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'warehouse-word'
wandb_run_name = 'gpt-warehouse'

dataset = 'warehouse'
gradient_accumulation_steps = 1
batch_size = 4
block_size = 128  # context of up to 128 previous *words*

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# device = 'dml'  # use 'dml' if you want to try DirectML
compile = False
