import os
import re
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("\n[APP] Loading model artifacts...")
with open(os.path.join(MODELS_DIR, "slang_classifier.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(MODELS_DIR, "slang_dictionary.pkl"), "rb") as f:
    slang_dict = pickle.load(f)

print(f"[APP] Loaded classifier, vectorizer, and {len(slang_dict)} slang entries.")
print("[APP] Ready!\n")
# Memisahkan dictionary untuk efisiensi eksekusi tokenisasi
multi_word_slang = {k: v for k, v in slang_dict.items() if ' ' in k}
single_word_slang = {k: v for k, v in slang_dict.items() if ' ' not in k}
def classify_word(word):
    """Classify a single word using the trained model."""
    X_vec = vectorizer.transform([word.lower()])
    prob = model.predict_proba(X_vec)[0]
    p_formal = prob[0]
    p_slang = prob[1]
    # Threshold konsisten 0.70
    pred = 1 if p_slang >= 0.70 else 0
    return pred, max(p_formal, p_slang), p_formal, p_slang
def correct_text(text):
    original_input = text
    results = []
    changes = []
    SLANG_THRESHOLD = 0.70 
    # 1. Eksekusi Multi-Word Slang (Contoh: "no cap") sebelum string di-split
    sorted_multi_words = sorted(multi_word_slang.keys(), key=lambda x: len(x.split()), reverse=True)
    for mw in sorted_multi_words:
        pattern = re.compile(r'\b' + re.escape(mw) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            corrected_phrase = multi_word_slang[mw]
            changes.append({
                "from": mw,
                "to": corrected_phrase,
                "confidence": 1.0,
                "p_slang": 1.0
            })
            text = pattern.sub(corrected_phrase, text)
    # 2. Tokenisasi sisa teks yang sudah steril dari frasa panjang
    tokens = text.split()
    corrected_tokens = []
    for token in tokens:
        # Pisahkan tanda baca di awal/akhir kata dasar
        match = re.match(r'^([^\w]*)(\w+)([^\w]*)$', token, re.UNICODE)
        if match:
            prefix = match.group(1)
            core = match.group(2)
            suffix = match.group(3)
            core_lower = core.lower()
            # Ambil probabilitas dari ML untuk kebutuhan log/UI dashboard
            _, _, p_formal, p_slang = classify_word(core_lower)
            # ── HIERARKI BARU: DICTIONARY OVERRIDE ──
            if core_lower in single_word_slang:
                corrected = single_word_slang[core_lower]
                if corrected.lower() != core_lower:
                    changes.append({
                        "from": core,
                        "to": corrected,
                        "confidence": 1.0,
                        "p_slang": round(float(p_slang), 4)
                    })
                corrected_tokens.append(prefix + corrected + suffix)
                results.append({
                    "original": core,
                    "is_slang": True,
                    "corrected": corrected,
                    "confidence": 1.0,
                    "p_formal": round(float(p_formal), 4),
                    "p_slang": round(float(p_slang), 4)
                })
            # ── FALLBACK 2: Jika kata tidak terdaftar di kamus resmi ──
            else:
                if p_slang >= SLANG_THRESHOLD: 
                    # ML mendeteksi sebagai slang baru (Unknown Slang)
                    corrected_tokens.append(token)
                    results.append({
                        "original": core,
                        "is_slang": True,
                        "corrected": core,  # Tetap apa adanya karena tidak ada di kamus normalisasi
                        "confidence": round(float(p_slang), 4),
                        "p_formal": round(float(p_formal), 4),
                        "p_slang": round(float(p_slang), 4)
                    })
                else:
                    # ML mendeteksi sebagai Kata Formal biasa (Contoh: "was", "so", "store")
                    corrected_tokens.append(token)
                    results.append({
                        "original": core,
                        "is_slang": False,
                        "corrected": core,
                        "confidence": round(float(p_formal), 4),
                        "p_formal": round(float(p_formal), 4),
                        "p_slang": round(float(p_slang), 4)
                    })
        else:
            # Tanda baca murni atau karakter non-kata
            corrected_tokens.append(token)
            results.append({
                "original": token,
                "is_slang": False,
                "corrected": token,
                "confidence": 1.0,
                "p_formal": 1.0,
                "p_slang": 0.0
            })

    final_corrected_text = " ".join(corrected_tokens)

    return {
        "original": original_input,
        "corrected": final_corrected_text,
        "total_changes": len(changes),
        "changes": changes,
        "word_details": results,
        "tokens_original": tokens,
        "tokens_corrected": corrected_tokens
    }
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/correct', methods=['POST'])
def correct():
    """Process text correction request via Web UI form/JSON."""
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        text = request.form.get('text', '')

    if not text.strip():
        return jsonify({
            'error': 'No text provided',
            'original': '',
            'corrected': '',
        })

    result = correct_text(text)
    return jsonify(result)


@app.route('/api/correct', methods=['GET'])
def api_correct():
    """REST API endpoint for external GET integration."""
    text = request.args.get('text', '')

    if not text.strip():
        return jsonify({
            'error': 'No text provided. Use ?text=your+text+here'
        })

    result = correct_text(text)
    return jsonify({
        'original': result['original'],
        'corrected': result['corrected'],\
        'changes': result['changes'],
        'total_changes': result['total_changes'],
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)