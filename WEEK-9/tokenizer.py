# MARİFETNAME METNİ UTF-8'E GÖRE ENCODE EDİLİR.
with open("WEEK-9/MARIFETNAME.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokens_marifetname = text.encode("utf-8")
tokens_marifetname = list(map(int, tokens_marifetname))

# CODE POINT VE BYTE SAYISI GÖSTERİM
turkce_karakterler_cumlesi = "Pijamalı hasta, yağız şoföre çabucak güvendi."
print(len(turkce_karakterler_cumlesi), len(turkce_karakterler_cumlesi.encode("utf-8")))
print(f"{'KARAKTER':<10} | {'UNICODE CP':<12} | {'BYTELAR':<15} | {'BYTE SAYISI'}")
print("-" * 55)
for c in turkce_karakterler_cumlesi:
    print(f"{c:<10} | {ord(c):<12} | {str(list(c.encode('utf-8'))):<15} | {len(c.encode('utf-8'))}")
# CODE POINT VE BYTE SAYISI GÖSTERİM


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

# İKİLİ SAYIM FONKSİYONU
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
# İKİLİ SAYIM FONKSİYONU


# MERGE FONKSİYONU (EN SIK TEKRARLANAN İKİLİLERİ MERGE EDER)
# aaabcdabeabe -> aaXcdXeXe
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
# MERGE FONKSİYONU (EN SIK TEKRARLANAN İKİLİLERİ MERGE EDER)


# print(sorted(((v, k) for k, v in stats.items()), reverse=True))

vocab_size = 1024
num_merges = vocab_size - 256
ids = list(tokens_marifetname) # Halihazırda list olan marifetname tokenlerini list metodu ile id'ye kopyalama işlemi gerçekleştirmiş olduk.

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    # print(f"merging {pair} into a new token {idx}: {stats[pair]} times encountered.")
    ids = merge(ids, pair, idx)
    merges[pair] = idx


# BPE SIKIŞTIRMA ORANI VE UZUNLUK KONTROLÜ
print("tokens length:", len(tokens_marifetname))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens_marifetname) / len(ids):.2f}X")
# BPE SIKIŞTIRMA ORANI VE UZUNLUK KONTROLÜ


# SÖZLÜK (VOCAB) OLUŞTURMA: 0-255 ARASI BYTE'LAR VE YENİ TOKEN'LAR
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
# SÖZLÜK (VOCAB) OLUŞTURMA: 0-255 ARASI BYTE'LAR VE YENİ TOKEN'LAR


# DECODE FONKSİYONU: TOKEN ID LİSTESİNİ UTF-8 METNE DÖNÜŞTÜRÜR
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
# DECODE FONKSİYONU: TOKEN ID LİSTESİNİ UTF-8 METNE DÖNÜŞTÜRÜR


# EĞİTİLEN İLK BİRLEŞTİRMELERİ EKRANA YAZDIRMA
print("--- STANDART BPE İLK 20 BİRLEŞTİRME ---")
for (p0, p1), idx in list(merges.items())[:20]:
    a = vocab[p0].decode("utf-8", errors="replace")
    b = vocab[p1].decode("utf-8", errors="replace")
    c = vocab[idx].decode("utf-8", errors="replace")
    print(f"{idx}: {a!r} + {b!r} -> {c!r}")
# EĞİTİLEN İLK BİRLEŞTİRMELERİ EKRANA YAZDIRMA


print(f"\nerrors=\"replace\" testi: {decode(bytes([120]))}, {decode(bytes([128]))}") # errors="replace" kodunun gereğini test etme.


# ENCODE FONKSİYONU: YENİ METNİ EĞİTİLMİŞ MERGES KURALLARINA GÖRE TOKEN ID LİSTESİNE DÖNÜŞTÜRÜR
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf"))) # idx'i en küçük olan pairi seçiyor.
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
# ENCODE FONKSİYONU: YENİ METNİ EĞİTİLMİŞ MERGES KURALLARINA GÖRE TOKEN ID LİSTESİNE DÖNÜŞTÜRÜR


# ENCODE - DECODE VE UNICODE TEST CASE'LERİ
print()
print(f"hello world encoding: {encode("hello world!")}")
print()
deneme_türkçe = "İstanbul’da hava bugün 23°C; ılık bir rüzgâr esiyor! 🌊 Şehrin eşsiz güzelliğini izlerken taze bir demli çay içmenin keyfi bambaşka... ☕✨ Şaşırdım mı? Hayır, çünkü şanlı tarihimiz ve kültürümüzle gurur duyuyoruz! 🇹🇷 Üç kağıtçı, paragöz veya vurdumduymaz insanlardan uzak; göğe bakıp uçurtma uçuran çocukların neşesiyle dolalım. 🪁🎈 (Not: Unicode testi için: ığüşiöç / IĞÜŞİÖÇ — ¡Hola! ¿Qué tal? 12345)"
print(f"decode(encode) metni: {decode(encode(deneme_türkçe))}")
print(deneme_türkçe == decode(encode(deneme_türkçe)))
print()
# ENCODE - DECODE VE UNICODE TEST CASE'LERİ


# REGEX PATTERN TANIMLARI VE KELİME BÖLME TESTLERİ
import regex as re
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
gpt4pat = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""")
print("--- REGEX TESTLERİ ---")
print(re.findall(gpt2pat, "Hello've world123 how's are you!!!?"))
print(re.findall(gpt2pat, "I've built a GPT Tokenizer. It's actually really     impressive!!!"))
print(re.findall(gpt2pat, "Berat'ın ismi Türkiye'de görülen en yaygın İSİMLERDEN değildir. Aksine, Mehmet'inki çok da yaygındır. 123456789"))
print(re.findall(gpt4pat, "Berat'ın ismi Türkiye'de görülen en yaygın İSİMLERDEN değildir. Aksine, Mehmet'inki çok da yaygındır. 123456789"))
print(re.findall(gpt4pat, "TÜRKİYE'DE, TÜRKİYE'NİN, TÜRKİYE'DEKİ, TÜRK'ÜN, TÜRK'SE, ingilizce'de, ingilizce'nin, ingilizce'deki, ingilizce'yse, Devlet'tir. 123456"))
print()
# REGEX PATTERN TANIMLARI VE KELİME BÖLME TESTLERİ

# MARİFETNAME METNİNİ GPT-4 PATTERN'I İLE CHUNK'LARA BÖLÜP BPE'Yİ HER PARÇADA AYRI ÇALIŞTIRMA
# Metin regex ile bağımsız parçalara ayrılır, farklı kelimelerin/sembollerin birbirine kaynaması engellenir.
chunks = [list(ch.encode("utf-8")) for ch in re.findall(gpt4pat, text)]

merges_regex = {}
for i in range(num_merges):
    counts = {}
    for tok in chunks:
        for pair in zip(tok, tok[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        break
    pair = max(counts, key=counts.get)
    idx = 256 + i
    chunks = [merge(tok, pair, idx) for tok in chunks]
    merges_regex[pair] = idx

# Regex tabanlı eğitilen yeni vocab yapısı
vocab_regex = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges_regex.items():
    vocab_regex[idx] = vocab_regex[p0] + vocab_regex[p1]

# ÖNCEKİ SONRAKİ KARŞILAŞTIRMASI
print("--- ÖNCEKİ STANDART BPE MERGES (İLK 10) ---")
for (p0, p1), idx in list(merges.items())[:10]:
    print(f"{idx}: {vocab[p0].decode("utf-8", errors="replace")!r} + {vocab[p1].decode("utf-8", errors="replace")!r} -> {vocab[idx].decode("utf-8", errors="replace")!r}")

print("--- REGEX CHUNK BPE MERGES (İLK 10) ---")
for (p0, p1), idx in list(merges_regex.items())[:10]:
    print(f"{idx}: {vocab_regex[p0].decode('utf-8', errors='replace')!r} + {vocab_regex[p1].decode('utf-8', errors='replace')!r} -> {vocab_regex[idx].decode('utf-8', errors='replace')!r}")
print()
# ÖNCEKİ SONRAKİ KARŞILAŞTIRMASI

# TIKTOKEN İLE TÜRKÇE İNGİLİZCE KARŞILAŞTIRMASI
import tiktoken
enc_cl100k = tiktoken.get_encoding("cl100k_base")

tr_paragraf = "Yapay zekâ modelleri metinleri doğrudan kelimeler hâlinde değil, token adı verilen küçük alt birimlere bölerek işler. Dillerin morfolojik yapıları ve hece düzenleri bu parçalama sürecini doğrudan belirler."
en_paragraf = "Artificial intelligence models do not process texts directly as whole words, but by breaking them down into smaller sub-units called tokens. The morphological structures and syllable patterns of languages directly determine this chunking process."

tr_kelime_sayisi = len(tr_paragraf.split())
en_kelime_sayisi = len(en_paragraf.split())

# cl100k_base ile tokenize etme
tr_cl100k_tokens = enc_cl100k.encode(tr_paragraf)
en_cl100k_tokens = enc_cl100k.encode(en_paragraf)

# Yeni fonksiyon yazmadan, ilk encode fonksiyonunu kullanarak tokenlaştırma
tr_custom_tokens = encode(tr_paragraf)

print(f"Türkçe Kelime: {tr_kelime_sayisi} | cl100k Token: {len(tr_cl100k_tokens)} | Kelime Başına Token: {len(tr_cl100k_tokens)/tr_kelime_sayisi:.2f}")
print(f"İngilizce Kelime: {en_kelime_sayisi} | cl100k Token: {len(en_cl100k_tokens)} | Kelime Başına Token: {len(en_cl100k_tokens)/en_kelime_sayisi:.2f}")
print(f"Türkçe Kelime: {tr_kelime_sayisi} | Custom Token: {len(tr_custom_tokens)} | Kelime Başına Token: {len(tr_custom_tokens)/tr_kelime_sayisi:.2f}")
