# NLP SLANG CLASSIFICATION & NORMALIZATION SYSTEM

## OVERALL SYSTEM DESCRIPTION
Sistem ini merupakan pipeline Natural Language Processing (NLP) yang digunakan untuk mengklasifikasikan kata sebagai slang atau formal, serta melakukan normalisasi slang menjadi bentuk formal menggunakan kombinasi Machine Learning dan Rule-Based System.

Sistem terdiri dari dua bagian utama:
1. TRAINING PIPELINE (main.ipynb)
2. DEPLOYMENT PIPELINE (app.py)

---

# =========================
# 1. TRAINING PIPELINE (main.ipynb)
# =========================

## 1. DATA LOADING
Pada tahap ini, sistem mengumpulkan data dari beberapa sumber:

- Slang datasets:
  - slang.csv
  - gen_zz_words.csv
  - genz_slang.csv
  - all_slangs.csv

- Formal dataset:
  - NLTK Words Corpus
  - Vocabulary modern tambahan (computer, internet, coding, dll)

Tujuan tahap ini adalah membangun dua kelas data utama:
- Slang (label 1)
- Formal (label 0)

---

## 2. DATA PREPROCESSING
Tahap ini bertujuan membersihkan data mentah agar siap diproses model.

Proses yang dilakukan:
- Lowercasing semua kata
- Menghapus whitespace berlebih menggunakan regex
- Filtering “poison words” (kata umum seperti is, are, you, the, dll)
- Validasi slang:
  - Maksimal 3 kata (agar tidak berupa kalimat panjang)
- Menghapus duplikasi data

Metode ini memastikan data bersih dan relevan untuk training model NLP.

---

## 3. DATASET CONSTRUCTION & LABELING
Pada tahap ini, data slang dan formal digabung menjadi satu dataset.

Labeling:
- Slang → 1
- Formal → 0

Hasil akhir berupa dataset supervised learning yang siap digunakan untuk training model klasifikasi.

---

## 4. DATA BALANCING
Karena jumlah data formal jauh lebih besar dibanding slang, dilakukan balancing.

Metode:
- Random shuffle data formal
- Downsampling formal agar jumlahnya sama dengan slang

Tujuan:
Menghindari bias model terhadap kelas formal.

---

## 5. TRAIN-TEST SPLIT
Dataset dibagi menjadi:
- 80% training data
- 20% testing data

Metode:
- Stratified split digunakan agar distribusi kelas tetap seimbang di train dan test set.

---

## 6. FEATURE EXTRACTION (TF-IDF + CHARACTER N-GRAM)
Ini adalah tahap konversi teks menjadi representasi numerik.

Metode yang digunakan:
- TF-IDF (Term Frequency - Inverse Document Frequency)
- Character-Level N-Gram (2–4)

Cara kerja:
Setiap kata dipecah menjadi pola karakter, contoh:
"rizz" → ["ri", "iz", "zz"]

Alasan penggunaan:
Slang sering memiliki variasi penulisan sehingga character-level representation lebih efektif dibanding word-level.

Output dari tahap ini adalah feature matrix yang digunakan untuk training model.

---

## 7. MODEL TRAINING (LOGISTIC REGRESSION)
Model yang digunakan adalah Logistic Regression.

Cara kerja dalam pipeline:
- Input: TF-IDF feature vector
- Proses: menghitung probabilitas kelas
- Output:
  - Probability slang (1)
  - Probability formal (0)

Keputusan model:
Jika p_slang lebih dominan → diklasifikasikan sebagai slang

Alasan penggunaan:
- Efisien untuk binary classification
- Stabil untuk high-dimensional sparse data (TF-IDF)

---

## 8. MODEL EVALUATION
Model dievaluasi menggunakan data testing dengan metrik:

- Accuracy → tingkat benar keseluruhan
- Precision → ketepatan prediksi slang
- Recall → kemampuan menangkap slang
- F1 Score → keseimbangan precision & recall
- Confusion Matrix → analisis kesalahan prediksi

Tujuan:
Mengukur performa generalisasi model terhadap data baru.

---

## 9. RULE-BASED SYSTEM (SLANG DICTIONARY)
Selain machine learning, sistem menggunakan dictionary-based normalization.

Isi dictionary:
- Dataset slang expansion
- Custom slang mapping manual

Contoh:
- rizz → charisma
- sus → suspicious
- frfr → for real

Fungsi dalam pipeline:
Digunakan sebagai prioritas utama sebelum model ML, untuk meningkatkan akurasi normalisasi slang yang sudah dikenal.

---

## 10. MODEL SAVING (DEPLOYMENT ARTIFACTS)
Setelah training selesai, tiga komponen disimpan:

- Logistic Regression Model → slang_classifier.pkl
- TF-IDF Vectorizer → tfidf_vectorizer.pkl
- Slang Dictionary → slang_dictionary.pkl

Tujuan:
Agar model tidak perlu training ulang saat digunakan di aplikasi.

---

# =========================
# 2. DEPLOYMENT PIPELINE (app.py)
# =========================

## 1. MODEL LOADING
Saat aplikasi dijalankan, sistem memuat:
- Trained Logistic Regression model
- TF-IDF vectorizer
- Slang dictionary

Tanpa tahap ini, sistem tidak dapat melakukan prediksi.

---

## 2. INPUT PROCESSING
User mengirimkan teks input.

Contoh:
"that guy is sus frfr"

Teks kemudian:
- Dibersihkan ringan
- Disiapkan untuk tokenisasi

---

## 3. MULTI-WORD SLANG HANDLING (RULE-BASED PRIORITY)
Sebelum tokenisasi, sistem mengecek multi-word slang.

Contoh:
- "no cap"
- "low key"

Jika ditemukan:
- langsung diganti ke makna formal
- dicatat sebagai perubahan

Tahap ini menggunakan dictionary-based system.

---

## 4. TOKENIZATION
Setelah multi-word slang diproses, teks dipecah menjadi token per kata.

Contoh:
"that guy is sus"
→ ["that", "guy", "is", "sus"]

---

## 5. WORD PROCESSING PIPELINE (HYBRID SYSTEM)

Setiap token diproses menggunakan dua pendekatan:

---

### (A) DICTIONARY CHECK (PRIORITAS 1)
Jika kata ada di slang dictionary:
- langsung dikonversi ke arti formal
- tidak masuk ke ML model

Contoh:
sus → suspicious

---

### (B) MACHINE LEARNING CLASSIFICATION (FALLBACK)

Jika kata tidak ada di dictionary:

Langkah:
1. TF-IDF vectorizer mengubah kata menjadi vector
2. Logistic Regression menghitung probabilitas

Output:
- p_slang
- p_formal

Decision rule:
Jika p_slang ≥ 0.70 → dianggap slang
Jika tidak → dianggap formal

---

## 6. TEXT RECONSTRUCTION
Setelah semua token diproses:
- kata digabung kembali
- menghasilkan kalimat final yang sudah dinormalisasi

---

## 7. OUTPUT RESPONSE
Sistem mengembalikan hasil dalam bentuk JSON:

- original text
- corrected text
- list perubahan kata
- detail probabilitas setiap kata

---

# =========================
# 3. EXAMPLE FLOW
# =========================

Input:
"that guy is sus frfr"

Step:
1. frfr → for real for real (dictionary)
2. sus → suspicious (dictionary)
3. remaining words → ML check

Output:
"that guy is suspicious for real for real"

---

# =========================
# 4. SUMMARY ARCHITECTURE
# =========================

TRAINING:
Data → Preprocessing → Labeling → Balancing → TF-IDF (Char N-Gram) → Logistic Regression → Evaluation → Save Model

DEPLOYMENT:
Load Model → Input Text → Rule-Based Check → Tokenization → ML Classification → Output Text
