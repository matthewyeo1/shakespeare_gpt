import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # no. of independent sequences processing in parallel
block_size = 256 # max context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200
n_embed = 384 # embedding dimension
n_head = 6 # number of attention heads
n_layer = 6 # number of layers in the transformer
dropout = 0.2 # dropout rate for regularization

torch.manual_seed(1337)

# Load text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create unique character list from text and mappings for encoding/decoding
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading function to generate batches
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random starting indices
    x = torch.stack([data[i:i+block_size] for i in ix]) # input batch
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # target batch
    x, y = x.to(device), y.to(device) # move tensors to the device (GPU/CPU)
    return x, y

# Function to estimate loss on train and validation data
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item() # record loss
        out[split] = losses.mean() # calculate average loss for the split
    model.train() # set model back to training mode
    return out

# Define one head of self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Create the key, query, and value linear transformations
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Create a lower-triangular matrix for masking future positions (in autoregressive settings)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    # Compute the attention scores ("affinities") and apply the mask
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute raw attention scores
        weight = q @ k.transpose(-2, -1) * C**-0.5    # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # mask future positions
        weight = F.softmax(weight, dim=-1)  # apply softmax to get probabilities
        weight = self.dropout(weight)  # apply dropout

        v = self.value(x)  # (B, T, head_size)
        out = weight @ v  # (B, T, head_size)
        return out

# Define the Multi-Head Attention mechanism (combining multiple Heads)
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Initialize multiple attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection to combine the heads' output
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all heads and apply projection
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embed)
        out = self.dropout(self.proj(out))  # (B, T, n_embed)
        return out

# Define a simple FeedForward network with two linear layers and ReLU activation
class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        # Define a simple feedforward network (2 layers with ReLU)
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # Expand the dimensions
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # Reduce back to original size
            nn.Dropout(dropout),  # Apply dropout for regularization
        )

    def forward(self, x):
        return self.net(x)

# Define a transformer block, consisting of self-attention followed by feedforward
class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head  # Size of each attention head
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-attention layer
        self.ffwd = FeedForward(n_embed)  # Feedforward network
        self.ln1 = nn.LayerNorm(n_embed)  # Layer normalization before attention
        self.ln2 = nn.LayerNorm(n_embed)  # Layer normalization before feedforward

    def forward(self, x):
        # Add residual connection around the self-attention and feedforward layers
        x = x + self.sa(self.ln1(x))  # Add residual connection for attention
        x = x + self.ffwd(self.ln2(x))  # Add residual connection for feedforward
        return x

# Define the Bigram Language Model (this is a very simple model for predicting sequences)
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Embedding layers for token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # Stack of transformer blocks (n_layer blocks)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)  # Final layer normalization
        self.lm_head = nn.Linear(n_embed, vocab_size)  # Linear layer for logits

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Look up token and position embeddings
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_embeddings + pos_embeddings # Add token and position embeddings
        x = self.blocks(x)    # Pass through transformer blocks
        logits = self.lm_head(x)  # Generate logits for vocabulary

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # Flatten logits
            targets = targets.view(B*T)   # Flatten targets
            loss = F.cross_entropy(logits, targets)  # Compute cross-entropy loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Generate new text by predicting next tokens
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Take last block_size tokens
            logits, loss = self(idx_cond)  # Get predictions
            logits = logits[:, -1, :]  # Focus on last token
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append next token to the sequence
        return idx

model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()  # Evaluate the model's performance
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data for training
    xb, yb = get_batch('train')

    # Evaluate loss and update model parameters
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))  # Generate 500 tokens
