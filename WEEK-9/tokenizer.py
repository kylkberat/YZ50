with open("WEEK-9/MARIFETNAME.txt", "r", encoding="utf-8") as f:
    text = f.read()

text2 = "Pijamalı hasta, yağız şoföre çabucak güvendi."
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))


print(len(text2), len(text2.encode("utf-8")))

print(f"{'KARAKTER':<10} | {'UNICODE CP':<12} | {'BYTELAR':<15} | {'BYTE SAYISI'}")
print("-" * 55)
for c in text2:
    print(f"{c:<10} | {ord(c):<12} | {str(list(c.encode('utf-8'))):<15} | {len(c.encode('utf-8'))}")



"""
Token olarak code pointleri neden kullanmıyoruz? Code pointler, her bir harfin karakterini idsini ifade eder 
ve 150 binden fazla karakter için code point bulunur. Eğer code point kullanacak olsaydık zaman zaman boyutu artırılan,
150 binden fazla karakter için bir embedding table oluşturacak ve devasa bir table'ı kontrol etmek zorunda kalacaktık.
Halbuki bu tablonun on binlerce satırı, o satıra ait karakter veride geçmediği için gradient alamayacak,
train edilemeyecek ve başlatıldığı halinde kalacaktı. 
Peki ya token olarak raw byte'ları kullansak? Raw byte'lar ascii tablosunda olan karakterlerin 1, kalanlarının 1'den fazla
byte ile tutulmasını sağlıyor olacak. Mesela "ş" harfi 2 byte veya 1 emoji 4 byte ile temsil edilecekti. Sabit bir 
block_size'da context uzun metinleri göremeyecek, yüksek bytelı karakterler sebebiyle aynı block_size içinde daha az 
metin görebilecekti. Bu durum, aynı metin boyutunu daha fazla maliyetle işlemek demek. Bu iki durumu dengede tutabilmek 
için BPE'yi tanıtıyoruz ortaya.
"""

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
# print(sorted(((v, k) for k, v in stats.items()), reverse=True))
top_pair = max(stats, key=stats.get)

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if (i < len(ids) - 1) and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    # print(f"merging {pair} into a new token {idx}: {stats[pair]} times encountered.")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

for (p0, p1), idx in list(merges.items())[:20]:
    a = vocab[p0].decode("utf-8", errors="replace")
    b = vocab[p1].decode("utf-8", errors="replace")
    c = vocab[idx].decode("utf-8", errors="replace")
    print(f"{idx}: {a!r} + {b!r} -> {c!r}")

print(decode(bytes([120])))

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

print()
print(f"hello world encoding: {encode("hello world!")}")
print()
deneme_türkçe = "İstanbul’da hava bugün 23°C; ılık bir rüzgâr esiyor! 🌊 Şehrin eşsiz güzelliğini izlerken taze bir demli çay içmenin keyfi bambaşka... ☕✨ Şaşırdım mı? Hayır, çünkü şanlı tarihimiz ve kültürümüzle gurur duyuyoruz! 🇹🇷 Üç kağıtçı, paragöz veya vurdumduymaz insanlardan uzak; göğe bakıp uçurtma uçuran çocukların neşesiyle dolalım. 🪁🎈 (Not: Unicode testi için: ığüşiöç / IĞÜŞİÖÇ — ¡Hola! ¿Qué tal? 12345)"
print(f"decode(encode) metni: {decode(encode(deneme_türkçe))}")
print(deneme_türkçe == decode(encode(deneme_türkçe)))
print()

import regex as re
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
gpt4pat = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""")
print(re.findall(gpt2pat, "Hello've world123 how's are you!!!?"))

print(re.findall(gpt2pat, "I've built a GPT Tokenizer. It's actually really     impressive!!!"))

print(re.findall(gpt2pat, "Berat'ın ismi Türkiye'de görülen en yaygın İSİMLERDEN değildir. Aksine, Mehmet'inki çok da yaygındır. 123456789"))
print(re.findall(gpt4pat, "Berat'ın ismi Türkiye'de görülen en yaygın İSİMLERDEN değildir. Aksine, Mehmet'inki çok da yaygındır. 123456789"))

print(re.findall(gpt4pat, "TÜRKİYE'DE, TÜRKİYE'NİN, TÜRKİYE'DEKİ, TÜRK'ÜN, TÜRK'SE, ingilizce'de, ingilizce'nin, ingilizce'deki, ingilizce'yse, Devlet'tir. 123456"))
print()

import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("   hello world!!!"))

enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("   hello world!!!"))


paragraf_tr = "Yapay zekâ modelleri metinleri doğrudan kelimeler halinde okumaz; bunun yerine onları token adı verilen küçük parçalara ayırarak işler. Türkçe gibi eklemeli dillerde eklerin çokluğu, genellikle aynı anlamı ifade etmek için İngilizceye kıyasla daha fazla token harcanmasına neden olur."
paragraf_en = "Artificial intelligence models do not read text directly as whole words; instead, they process it by breaking it down into smaller pieces called tokens. In agglutinative languages like Turkish, the abundance of suffixes often results in consuming more tokens compared to English to convey the same meaning."
print(len(enc.encode(paragraf_tr)), len(paragraf_tr.split()), len(encode(paragraf_tr)))
print(len(enc.encode(paragraf_en)), len(paragraf_en.split()))
