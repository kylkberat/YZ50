import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

# TRANSFORMERS'I YÜKLE VE TENSORLERİN İSİMLERİNİ YAZDIR
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)

# PARAMETRE SAYISINI VİRGÜLLÜ YAZ
n_params = sum(p.numel() for p in model_hf.parameters())
print(f"toplam parametre: {n_params:,}")


# GRAFİKLERİ ÇİZ ------------------
wpe = sd_hf["transformer.wpe.weight"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
axes[0].imshow(wpe, cmap="gray", aspect="auto")
axes[0].set_xlabel("embedding dim")
axes[0].set_ylabel("position")
 
for d in (150, 200, 250):
    axes[1].plot(wpe[:, d], label=f"dim {d}")
axes[1].set_xlabel("position")
axes[1].legend()

""" 
plt.tight_layout()
plt.savefig("wpe.png", dpi=120)
plt.show()
"""
# GRAFİKLERİ ÇİZ ------------------

# GENERATOR TESTİ -----------------
from transformers import pipeline, set_seed
generator = pipeline("text-generation", model="gpt2")
set_seed(42)
print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5))
