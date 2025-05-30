import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device}")
n_embed = 384
n_heads = 6
eval_iters = 200
dropout=0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./study/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x (B, T, C)
        output = torch.concat([head(x) for head in self.heads], dim = -1) # (B, T, head_size * num_heads)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class Head(nn.Module):
    # One single head of self attention
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril==0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, head_size)

        out = wei @ v # (B, T, head_size) 
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU()
        )
        self.proj = nn.Linear(n_embed*4, n_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.net(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size) # Communication
        self.ffwd = FeedForward(n_embed) # Compuation
        self.ln1 = nn.LayerNorm(n_embed) # Layer norm 1
        self.ln2 = nn.LayerNorm(n_embed) # Layer norm 22

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed).to(device)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed).to(device)
        self.blocks = nn.Sequential(
            Block(n_embed, n_heads=n_heads),
            Block(n_embed, n_heads=n_heads),
            Block(n_embed, n_heads=n_heads),
            nn.LayerNorm(n_embed)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        # idx and target has dimension (B, T)
        _, T = idx.shape
        token_embed = self.token_embedding_table(idx) # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(0,T).to(device)) # (T, C)
        x = token_embed + pos_embed # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            losses = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            losses = F.cross_entropy(input=logits, target=targets)

        return logits, losses
    
    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond) # (B, T, C)
            logits = logits[:, -1, :] # (B, C) We only want the last timestamp prediction
            probs = F.softmax(logits, dim=1) # (B, C)
            next_idx = torch.multinomial(input=probs, num_samples=1) # 1 sample for the distribution
            idx = torch.concat([idx, next_idx], dim=1)
        return idx

model = GPT(vocab_size=vocab_size).to(device)
optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)

x, y = get_batch("val")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model training
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(torch.tensor([[ 1, 39, 45, 39, 47, 52,  1, 57, 46, 39, 50, 50,  1, 57, 43, 43,  1, 46,
        43, 56,  8,  1, 27,  1, 58, 46, 53, 59,  1, 51, 47, 52, 43,  1, 46, 43,
        47, 56,  0, 27, 44,  1, 26, 39, 54, 50, 43, 57,  1, 39, 52, 42,  1, 53,
        44,  1, 25, 47, 50, 39, 52,  6,  1, 61, 46, 39, 58,  1, 57, 58, 56, 39,
        52, 45, 43,  1, 44, 47, 57, 46,  0, 20, 39, 58, 46,  1, 51, 39, 42, 43,
         1, 46, 47, 57,  1, 51, 43, 39, 50,  1, 53, 52,  1, 58, 46, 43, 43, 12,
         0,  0, 18, 30, 13, 26, 15, 21, 31, 15, 27, 10,  0, 31, 47, 56,  6,  1,
        46, 43,  1, 51, 39, 63,  1, 50, 47, 60, 43, 10,  0, 21,  1, 57, 39, 61,
         1, 46, 47, 51,  1, 40, 43, 39, 58,  1, 58, 46, 43,  1, 57, 59, 56, 45,
        43, 57,  1, 59, 52, 42, 43, 56,  1, 46, 47, 51,  6,  0, 13, 52, 42,  1,
        56, 47, 42, 43,  1, 59, 54, 53, 52,  1, 58, 46, 43, 47, 56,  1, 40, 39,
        41, 49, 57, 11,  1, 46, 43,  1, 58, 56, 53, 42,  1, 58, 46, 43,  1, 61,
        39, 58, 43, 56,  6,  0, 35, 46, 53, 57, 43,  1, 43, 52, 51, 47, 58, 63,
         1, 46, 43,  1, 44, 50, 59, 52, 45,  1, 39, 57, 47, 42, 43,  6,  1, 39,
        52, 42,  1, 40]]).to(device), max_new_tokens=500)[0].tolist()))
print()