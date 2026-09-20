import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import random

words = open('WEEK-4/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

g = torch.Generator().manual_seed(2147483647)
block_size = 5
embed_dim = 35
hid_neuron_cnt = 35

def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for char in w + '.':
            ix = stoi[char]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtrain, Ytrain = build_dataset(words[:n1], block_size)
Xdev, Ydev = build_dataset(words[n1:n2], block_size)
Xtest, Ytest = build_dataset(words[n2:], block_size)

def cmp(s, dt, t):
    exact = torch.all(dt == t.grad).item()
    appr = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | exact: {str(exact):5s} | approximate: {str(appr):5s} | maxdiff = {maxdiff}")

C = torch.randn((len(stoi), embed_dim), generator=g)
W1 = torch.randn((block_size * embed_dim, hid_neuron_cnt), generator=g) * ( (5/3) / ((block_size * embed_dim) ** 0.5) )
b1 = torch.randn(hid_neuron_cnt, generator=g) * 0
W2 = torch.randn((hid_neuron_cnt, len(stoi)), generator=g) * 0.1
b2 = torch.randn(len(stoi), generator=g) * 0.1 

bngain = torch.ones((1, hid_neuron_cnt))
bnbias = torch.zeros((1, hid_neuron_cnt))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True


# minibatch
n = 32
ix = torch.randint(0, Xtrain.shape[0], (n,))

# forward pass
emb = C[Xtrain[ix]]
embcat = emb.view(emb.shape[0], -1)
hprebn = embcat @ W1 + b1
bnmeani = 1/n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = 1 / (n-1) * (bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = torch.tanh(hpreact)
logits = h @ W2 + b2
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Ytrain[ix]].mean()

# backward pass
for p in parameters:
    p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
            norm_logits, logit_maxes, logits, h, hpreact, bnraw,
            bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
            embcat, emb]:
    t.retain_grad()
loss.backward()

print(emb.shape, C.shape, Xtrain[ix].shape)
# print(f"hprebn {hprebn.shape}\nembcat {embcat.shape}\nW1 {W1.shape}\nb1 {b1.shape}")
# print(f"bndiff {bndiff.shape}\nbnmeani {bnmeani.shape}\nhprebn {hprebn.shape}")
# print(f"{bnvar.shape}\n{bndiff2.shape}\n{bndiff.shape}")
# print(f"{bnraw.shape}\n{bnvar_inv.shape}\n{bndiff.shape}")
# print(f"hpre {hpreact.shape}\nbngain {bngain.shape}\nbnraw {bnraw.shape}\nbnbias {bnbias.shape}")
# print(f"{logits.shape}\n{h.shape}\n{W2.shape}\n{b2.shape}")

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Ytrain[ix]] = -1.0/n
dprobs = (1.0 / probs) * dlogprobs
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
dcounts_sum = -dcounts_sum_inv * (counts_sum ** -2) 
dcounts = (dprobs * counts_sum_inv) + (dcounts_sum * torch.ones_like(counts))
dnorm_logits = dcounts * norm_logits.exp()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits = dnorm_logits.clone()
dlogits += (F.one_hot(logits.max(1).indices, num_classes=logits.shape[1])) * dlogit_maxes
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)
dhpreact = (1 - h ** 2) * dh
dbnbias = dhpreact.sum(0, keepdim=True)
dbngain = (dhpreact * (bnraw)).sum(0, keepdim=True)
dbnraw = dhpreact * bngain
dbndiff = dbnraw * bnvar_inv
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)
dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
dbndiff2 = (1.0 / (n-1)) * torch.ones_like(bndiff2) * dbnvar
dbndiff += dbndiff2 * 2 * bndiff
dbnmeani = (-1 * dbndiff).sum(0, keepdim=True)
dhprebn = dbndiff.clone() + 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for k in range(Xtrain[ix].shape[0]):
    for l in range(Xtrain[ix].shape[1]):
        ix2 = Xtrain[ix][k, l]
        dC[ix2] += demb[k, l]

# cmp("logprobs", dlogprobs, logprobs)
# cmp("probs", dprobs, probs)
# cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)
# cmp("counts_sum", dcounts_sum, counts_sum)
# cmp("counts", dcounts, counts)
# cmp('norm_logits', dnorm_logits, norm_logits)
# cmp('logit_maxes', dlogit_maxes, logit_maxes)
# cmp('logits', dlogits, logits)
# cmp('h', dh, h)
# cmp('W2', dW2, W2)
# cmp('b2', db2, b2)
# cmp('hpreact', dhpreact, hpreact)
# cmp('bngain', dbngain, bngain)
# cmp('bnbias', dbnbias, bnbias)
# cmp('bnraw', dbnraw, bnraw)
# cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
# cmp('bnvar', dbnvar, bnvar)
# cmp('bndiff2', dbndiff2, bndiff2)
# cmp('bndiff', dbndiff, bndiff)
# cmp('bnmeani', dbnmeani, bnmeani)
# cmp('hprebn', dhprebn, hprebn)
# cmp('embcat', dembcat, embcat)
# cmp('W1', dW1, W1)
# cmp('b1', db1, b1)
# cmp('emb', demb, emb)
# cmp('C', dC, C)
