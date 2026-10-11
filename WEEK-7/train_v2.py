import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 5000
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
embed_dim = 32
head_size = 16

torch.manual_seed(1337)

with open("WEEK-7/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim) # t h a t / h a t t
        self.position_embedding_table = nn.Embedding(block_size, embed_dim) # (8, 32)
        self.sa_head = Head(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        tok_emb = self.token_embedding_table(idx) # (B,T, embed_dim)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
    
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter == max_iters - 1:
        losses = estimate_loss()
        print(f"train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# WEIGHTED SUM GÖSTERMEK İÇİN ÖRNEK SİSTEM

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print("a = ")
print(a)
print("b = ")
print(b)
print("c = ")
print(c)

# WEIGHTED SUM GÖSTERMEK İÇİN ÖRNEK SİSTEM


# ÜÇ FARKLI METOD HESABI İÇİN SİSTEM KURULUMU
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

    # FOR DÖNGÜSÜ
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)
        if b == 0:
            print(f"xprev {t}. adımda:\n {xprev}\n")
            print(f"xbow {t}. adımda:\n {xbow[b, t]}\n\n")


    # LOWER TRIANGULAR MATRIX
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
# print(wei)
xbow2 = wei @ x # Buradaki shapeler: (B, T, T) @ (B, T, C) ---> (B, T, C)
print(torch.allclose(xbow, xbow2, atol=1e-6)) # atol: absolute tolerance: floating point hatası yüzünden 1e-7 gibi bir kayma hatası oluşuyor.


    # SOFTMAX
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
print(wei)
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3, atol=1e-6))
# Softmax'in lower triangular'a tercih edilmesinin sebebi attention'da raw logits'leri hesapladıktan sonra sampling için gerekli
# softmax'i yaparken exponential function kullanılıyor, ağırlıkların ölçekleri ve farkları büyüdükçe logits dengesizleşmeye başlıyor.

 