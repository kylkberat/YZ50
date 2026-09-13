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
    # print(X.shape, Y.shape)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtrain, Ytrain = build_dataset(words[:n1], block_size)
Xdev, Ydev = build_dataset(words[n1:n2], block_size)
Xtest, Ytest = build_dataset(words[n2:], block_size)


C = torch.randn((len(stoi), embed_dim), generator=g)
# shape of C[X] = (32, 3, 2) = (characters, block_size, embed_dim)
# emb = C[X]

W1 = torch.randn((block_size * embed_dim, hid_neuron_cnt), generator=g)
b1 = torch.randn(hid_neuron_cnt, generator=g)

"""
h = torch.tanh(emb.view(-1, embed_dim * block_size) @ W1 + b1)
# emb.view(32, 6) == torch.cat(torch.unbind(emb, 1), 1)
"""

W2 = torch.randn((hid_neuron_cnt, len(stoi)), generator=g)
b2 = torch.randn(len(stoi), generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True


"""
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
loss = -prob[torch.arange(X.shape[0]), Y].log().mean() # torch.arange kısmı satırı, Y ise sütunu getiriyor.
"""


"""
Bütün bu ara basamaklarla loss hesaplamak yerine F.cross_entropy ile
doğrudan loss hesabı yapılabilir. Bunun getirileri:
1) Her ara basamakta yeni tensor oluşturulmaz.
2) Backward pass kolayca yapılabilir.
3) Ara basamaklar tek bir batch halinde fonksiyonun içindedir.
4) .exp()'e 100 gibi fazla pozitif logits'ler verilemez. inf'den dolayı nan'a götürür.
cross_entropy bunu logitslerden max olanını çıkararak çözer.
*([-5, -2, 0, 100] - 100)
"""

# lr_exp = torch.linspace(-3, 0, 1000)
# lrs = 10**lr_exp
# lri, lossi = [], []
for i in range(30000):

    # minibatch
    ix = torch.randint(0, Xtrain.shape[0], (32,))

    # forward pass
    emb = C[Xtrain[ix]]
    h = torch.tanh(emb.view(-1, embed_dim * block_size) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytrain[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    learning_rate = 10 ** (-0.95)
    for p in parameters:
        p.data -= learning_rate * p.grad

    # tracking
    # lri.append(lr_exp[i])
    # lossi.append(loss.item())

#plt.plot(lri, lossi)
#plt.show()
print(loss.item())

def tr_dev(X, Y, parameters, number):
    emb = parameters[0][X]
    h = torch.tanh(emb.view(-1, number) @ parameters[1] + parameters[2])
    logits = h @ parameters[3] + parameters[4]
    loss = F.cross_entropy(logits, Y)
    return loss

loss_train = tr_dev(Xtrain, Ytrain, parameters, (embed_dim * block_size))
loss_dev = tr_dev(Xdev, Ydev, parameters, (embed_dim * block_size))
print(loss_train.item())
print(loss_dev.item())

""" -----------  harf kümesi grafik kodu  -----------
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
plt.grid("minor")
plt.show()
"""

def sampling(parameters, block_size, itos, sample_count, generator_code):
    C, W1, b1, W2, b2 = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]

    for _ in range(sample_count):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=generator_code).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print("".join(itos[i] for i in out))

generator_code = torch.Generator().manual_seed(2147483647 + 10)
sampling(parameters, block_size, itos, 20, generator_code)

"""with open("WEEK-4/losses.txt", "a", encoding="utf-8") as f:
    f.write(
        f"HYPERPARAMETERS\n"
        f"{'-' * 30}\n"
        f"block_size: {block_size}\n"
        f"embed_dim: {embed_dim}\n"
        f"hid_neuron_cnt: {hid_neuron_cnt}\n"
        f"train loss: {loss_train.item():.4f}\n"
        f"dev loss: {loss_dev.item():.4f}\n"
        f"{'-' * 30}\n"
    )"""