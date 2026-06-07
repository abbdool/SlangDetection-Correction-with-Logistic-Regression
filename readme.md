📌 🔥 1. TRAINING PIPELINE
🟦 1. Data Loading
Sumber data:
4 dataset slang (CSV):
slang.csv
gen_zz_words.csv
genz_slang.csv
all_slangs.csv
NLTK corpus:
words.words() → formal English words
tambahan vocabulary modern
🟦 2. Data Cleaning & Preprocessing
Fungsi:

Membersihkan dan menyaring data agar valid untuk training.

Proses:
lowercase semua kata
hapus whitespace berlebih
hapus “poison words” (kata umum seperti: is, are, you, the, dll)
validasi slang:
maksimal 3 kata (biar bukan kalimat)
remove duplicate dengan set()
🟦 3. Dataset Construction
Slang dataset:
gabungan semua slang list
Formal dataset:
NLTK words + modern vocabulary
lalu difilter agar tidak mengandung slang
🟦 4. Data Balancing
Tujuan:

Menghindari model bias ke salah satu kelas.

Cara:
shuffle formal words
ambil jumlah formal = jumlah slang
🟦 5. Labeling
Output dataset:
Word	Label
slang	1
formal	0
🟦 6. Train-Test Split
Metode:
80% training
20% testing
stratified split (biar distribusi class tetap sama)
🟦 7. Feature Extraction (TF-IDF Character N-Gram)
Tools:
TfidfVectorizer
Konfigurasi:
analyzer = character
ngram_range = (2,4)
Fungsi:
Mengubah kata menjadi representasi angka berdasarkan pola karakter.

Contoh:
"rizz" → "ri", "iz", "zz", dll
🟦 8. Model Training
Algoritma:
Logistic Regression
Fungsi:
belajar membedakan slang vs formal berdasarkan TF-IDF feature
🟦 9. Model Evaluation
Metrics:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Tujuan:
Mengukur performa model di data test

🟦 10. Slang Dictionary Construction (Rule-Based System)
Isi:
slang dari dataset (acronym → expansion)
custom dictionary manual
Fungsi:
mempercepat normalisasi slang tanpa ML
meningkatkan akurasi sistem
🟦 11. Model Saving (Deployment Preparation)
Disimpan pakai pickle:
slang_classifier.pkl → model Logistic Regression
tfidf_vectorizer.pkl → feature extractor
slang_dictionary.pkl → kamus slang
📌 🚀 2. DEPLOYMENT / INFERENCE PIPELINE
🟩 1. Load Model & Assets

Saat aplikasi start:

Load Logistic Regression model
Load TF-IDF vectorizer
Load slang dictionary
Hasil:

Sistem siap menerima input user

🟩 2. Preprocessing Input User

Input contoh:

"that guy is sus frfr"

Tidak dilakukan training lagi, hanya:

parsing text
tokenization
regex cleaning ringan
🟩 3. Multi-Word Slang Detection (RULE BASED FIRST)
Contoh:
"no cap"
"low key"
Proses:
sistem cek phrase dulu sebelum split kata
jika ditemukan:
langsung diganti ke formal meaning
dicatat sebagai perubahan
🟩 4. Tokenization

Setelah multi-word slang diproses:

teks di-split jadi token per kata

Contoh:

["that", "guy", "is", "sus"]
🟩 5. Word Classification (HYBRID SYSTEM)

Untuk setiap kata:

🔹 Step A — Dictionary Check (PRIORITY 1)

Jika kata ada di slang dictionary:

sus → suspicious
rizz → charisma

➡️ langsung diganti (RULE-BASED OUTPUT)

🔹 Step B — ML Prediction (FALLBACK)

Jika tidak ada di dictionary:

Proses:
TF-IDF vectorizer transform kata
Logistic Regression predict probability

Output:

p_slang
p_formal
Decision rule:
if p_slang >= 0.70:
    dianggap slang
else:
    dianggap formal
🟩 6. Reconstruction Output Text

Setelah semua token diproses:

kata digabung kembali
menghasilkan teks final
🟩 7. Output Response (API / UI)

Output JSON:

{
  "original": "...",
  "corrected": "...",
  "changes": [...],
  "word_details": [...]
}
📌 🔗 3. CONTOH END-TO-END FLOW
Input:
"that guy is sus frfr"
Step 1: Multi-word slang
frfr → for real for real
Step 2: Tokenize
["that", "guy", "is", "sus", "for", "real", "for", "real"]
Step 3: Dictionary check
sus → suspicious
Step 4: ML check
that, guy, is → formal
for, real → formal
Output:
"that guy is suspicious for real for real"
📌 🧠 4. HUBUNGAN MAIN.IPYNB VS APP.PY
🔥 main.ipynb (TRAINING)
bikin otak model
belajar dari dataset
menghasilkan:
model
vectorizer
dictionary
🚀 app.py (INFERENCE)
pakai otak yang sudah jadi
tidak training lagi
hanya:
predict
normalize
return hasil
