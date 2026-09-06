import torch
import matplotlib.pyplot as plt
from torch.nn import functional as Func


words = open('WEEK-3/names.txt', 'r').read().splitlines()

# ------------------------------------------------ TASK 1 ----------------------------------------------
# --------------------- Dict ile ------------------------
b = dict()
for word in words[:3]:
    chs = ['<S>'] + list(word) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1  # b.get(bigram) = b[bigram] ama b.get(bigram, 0) yazdığımızda yoksa default 0 koy diyoruz. defaultdict gibi
# -------------------------------------------------------


# ----------------------Torch ile -----------------------
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)

for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis('off')
plt.show()

# --------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


# ------------------------------------------------- TASK 2 ----------------------------------------------
P = (N+1).float()
P /= P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)
for _ in range(10):
    out = []
    ix = 0
    while True:

        p = P[ix]
        # p = N[ix].float() verimi artırmak adına bunları döngü dışına taşıyoruz.
        # p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    # print(''.join(out))
# -------------------------------------------------------------------------------------------------------


# ---------------------------------------------- TASK 3 -------------------------------------------------
log_likelihood = 0.0
n = 0
for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

#print(f"{log_likelihood=}")        
nll = -log_likelihood
#print(f"{nll=}")
print(f"{nll/n=}")
# -------------------------------------------------------------------------------------------------------


# -------------------------------------------- TASK 4 ---------------------------------------------------
xs = []
ys = []
for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((len(stoi), len(stoi)), generator=g, requires_grad=True)

for _ in range(300):
    xenc = Func.one_hot(xs, num_classes=len(stoi)).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item())

    W.grad = None
    loss.backward()

    W.data -= 50 * W.grad