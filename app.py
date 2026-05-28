"""
=============================================================
  ETS 2 - Aplikasi Speech-to-Text & Text-to-Speech
  Mata Kuliah: IFB-306 Pengenalan Ucapan & Teks ke Ucapan
  NRP   : 152023013
  Nama  : Cleosa Zelda Avrillya
  Metode: MFCC + DTW (Dynamic Time Warping)
=============================================================
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import librosa
import os, json, pickle, io, base64, wave, struct, math, tempfile
from scipy.spatial.distance import euclidean

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ── Vocabulary ──────────────────────────────────────────────
VOCAB = {
    # Bilangan asli 0–10
    "nol":      "0",   "satu":     "1",   "dua":      "2",
    "tiga":     "3",   "empat":    "4",   "lima":     "5",
    "enam":     "6",   "tujuh":    "7",   "delapan":  "8",
    "sembilan": "9",   "sepuluh":  "10",
    # Operator aritmatika
    "tambah":   "+",   "kurang":   "-",   "kali":     "*",   "bagi":   ":",
    # Identitas – NRP & nama
    "cleosa":   "cleosa",   "zelda":  "zelda",   "avrillya": "avrillya",
}
LABEL_TO_WORD = {v: k for k, v in VOCAB.items()}

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
MODELS_DIR     = os.path.join(os.path.dirname(__file__), "models")
AUDIO_DIR      = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,     exist_ok=True)
os.makedirs(AUDIO_DIR,      exist_ok=True)

TEMPLATE_FILE = os.path.join(MODELS_DIR, "templates.pkl")

# ── NRP / Nama expansion ─────────────────────────────────────
NRP_DIGITS = list("152023013")   # digit chars
NRP_WORDS  = {
    "0": "nol",  "1": "satu", "2": "dua",  "3": "tiga",
    "4": "empat","5": "lima", "6": "enam", "7": "tujuh",
    "8": "delapan","9": "sembilan",
}
DIGIT_TO_WORD = NRP_WORDS
NAMA_WORDS = ["cleosa", "zelda", "avrillya"]

# ─────────────────────────────────────────────────────────────
#  SIGNAL PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────

def preprocess_audio(y, sr, target_sr=16000):
    """Resample, mono, normalize."""
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.effects.trim(y, top_db=20)[0]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y, target_sr

def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extract MFCC features (13 coefficients) as used in course material.
    Returns matrix shape (frames, n_mfcc).
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=160,   # 10 ms at 16 kHz
        win_length=400,   # 25 ms window
        n_mels=26,
        fmax=8000,
    )
    # Delta features for better discrimination
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])  # (39, frames)
    return features.T   # (frames, 39)

def dtw_distance(seq_a, seq_b, radius=10):
    """
    DTW distance with Sakoe-Chiba band.
    seq_a, seq_b: numpy arrays of shape (T, features).
    """
    n, m = len(seq_a), len(seq_b)
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0][0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end   = min(m, i + radius)
        for j in range(j_start, j_end + 1):
            cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            D[i][j] = cost + min(D[i-1][j], D[i][j-1], D[i-1][j-1])

    dist = D[n][m]
    # Normalize by path length
    return dist / (n + m)


# ─────────────────────────────────────────────────────────────
#  TEMPLATE STORE
# ─────────────────────────────────────────────────────────────

def load_templates():
    if os.path.exists(TEMPLATE_FILE):
        with open(TEMPLATE_FILE, "rb") as f:
            return pickle.load(f)
    return {}   # {word: [mfcc_matrix, ...]}

def save_templates(templates):
    with open(TEMPLATE_FILE, "wb") as f:
        pickle.dump(templates, f)

def add_template(word, mfcc_matrix):
    templates = load_templates()
    if word not in templates:
        templates[word] = []
    templates[word].append(mfcc_matrix)
    save_templates(templates)

def get_stats():
    templates = load_templates()
    stats = {}
    for word in VOCAB.keys():
        stats[word] = len(templates.get(word, []))
    return stats


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           vocab=list(VOCAB.keys()),
                           nrp="152023013",
                           nama="Cleosa Zelda Avrillya")

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/api/record", methods=["POST"])
def api_record():
    """
    Receive recorded audio + label, extract MFCC, save as template.
    Body: multipart/form-data  {audio: file, word: str}
    """
    import soundfile as sf
    try:
        word  = request.form.get("word", "").lower().strip()
        audio = request.files.get("audio")
        if not word or word not in VOCAB:
            return jsonify({"error": f"Kata tidak dikenal: {word}"}), 400
        if not audio:
            return jsonify({"error": "File audio tidak ditemukan"}), 400

        raw = audio.read()
        
        # Simpan berkas sementara berformat .webm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            # Menggunakan librosa untuk membaca file webm sementara
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
        except Exception as sound_err:
            return jsonify({
                "error": f"Gagal membaca format WebM. Pastikan FFMPEG terinstal di perangkat Anda atau gunakan browser modern (Chrome/Edge). Detail: {str(sound_err)}"
            }), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        y, sr = preprocess_audio(y, sr)
        mfcc  = extract_mfcc(y, sr)

        if len(mfcc) < 5:
            return jsonify({"error": "Rekaman terlalu pendek, coba lagi"}), 400

        add_template(word, mfcc)

        # FIX UTAMA: Simpan berkas .wav murni menggunakan soundfile (bukan biner webm langsung)
        tts_path = os.path.join(AUDIO_DIR, f"{word}.wav")
        if not os.path.exists(tts_path):
            sf.write(tts_path, y, sr, format='WAV', subtype='PCM_16')

        count = len(load_templates().get(word, []))
        return jsonify({
            "success": True,
            "word": word,
            "count": int(count),
            "frames": int(len(mfcc)),
        })
    except Exception as e:
        return jsonify({"error": f"Crash internal pada record: {str(e)}"}), 500


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    Receive audio, segment it into words, and recognize each segment 
    with full top-3 candidates tracking for each word.
    """
    try:
        audio = request.files.get("audio")
        if not audio:
            return jsonify({"error": "File audio tidak ada"}), 400

        raw = audio.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        if y.ndim > 1:
            y = y.mean(axis=1)

        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        templates = load_templates()
        if not templates:
            return jsonify({"error": "Belum ada template. Silakan rekam data latih terlebih dahulu!"}), 400

        intervals = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
        
        if len(intervals) == 0:
            intervals = np.array([[0, len(y)]])

        recognized_words = []
        recognized_symbols = []
        total_distance = 0
        segment_logs = []  # Menyimpan riwayat ranking per segmen kata

        for idx, (start, end) in enumerate(intervals):
            y_segment = y[start:end]
            
            if len(y_segment) < 1600: 
                continue
                
            y_segment = librosa.effects.trim(y_segment, top_db=20)[0]
            if len(y_segment) == 0:
                continue
            y_segment = y_segment / (np.max(np.abs(y_segment)) + 1e-6)

            segment_mfcc = extract_mfcc(y_segment, sr)
            if len(segment_mfcc) < 3:
                continue

            best_word = None
            best_dist = float("inf")
            distances = {}

            # Hitung jarak ke SEMUA template untuk diranking
            for word, mfcc_list in templates.items():
                word_dists = [dtw_distance(segment_mfcc, tmpl) for tmpl in mfcc_list]
                min_dist = min(word_dists)
                distances[word] = round(min_dist, 4)
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_word = word

            if best_word:
                recognized_words.append(best_word)
                recognized_symbols.append(VOCAB.get(best_word, "?"))
                total_distance += best_dist
                
                # Ambil 3 teratas untuk segmen kata ini
                top3_for_segment = sorted(distances.items(), key=lambda x: x[1])[:3]
                segment_logs.append({
                    "segment_index": idx + 1,
                    "winner": best_word,
                    "top3": top3_for_segment
                })

        if not recognized_words:
            return jsonify({"error": "Tidak ada kata yang berhasil dikenali dengan jelas."}), 400

        final_recognized = " ".join(recognized_words)
        final_symbol = " ".join(recognized_symbols)
        avg_distance = round(total_distance / len(recognized_words), 4)

        return jsonify({
            "recognized": final_recognized,
            "symbol":     final_symbol,
            "distance":   avg_distance,
            "confidence": 90.0,
            "segment_logs": segment_logs  # Kita kirim data log ranking per kata ke frontend
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/tts", methods=["POST"])
def api_tts():
    """
    TTS: given text, return audio sequence.
    Handles: numbers, operators, NRP, nama.
    """
    data = request.get_json()
    text = data.get("text", "").strip().lower()

    words_to_speak = []

    # Special keywords
    if text in ("nrp", "152023013"):
        words_to_speak = [NRP_WORDS[d] for d in NRP_DIGITS]
    elif text in ("nama", "cleosa zelda avrillya"):
        words_to_speak = NAMA_WORDS
    elif text in ("identitas", "perkenalan"):
        words_to_speak = (
            [NRP_WORDS[d] for d in NRP_DIGITS] + NAMA_WORDS
        )
    else:
        # Parse token by token
        for token in text.split():
            if token in VOCAB:
                words_to_speak.append(token)
            elif token.isdigit():
                # digit 0-10
                num = int(token)
                inv = {v: k for k, v in VOCAB.items()}
                w = inv.get(str(num)) or NRP_WORDS.get(token)
                if w:
                    words_to_speak.append(w)
            elif token in ("+", "-", "*", ":"):
                sym_map = {"+": "tambah", "-": "kurang",
                           "*": "kali",  ":": "bagi"}
                words_to_speak.append(sym_map[token])

    if not words_to_speak:
        return jsonify({"error": "Tidak ada kata yang dikenali dalam teks", "words": []}), 400

    # Check which audio files exist
    available = []
    missing   = []
    for w in words_to_speak:
        path = os.path.join(AUDIO_DIR, f"{w}.wav")
        if os.path.exists(path):
            available.append(w)
        else:
            missing.append(w)

    audio_urls = [f"/static/audio/{w}.wav" for w in words_to_speak]

    return jsonify({
        "words":      words_to_speak,
        "audio_urls": audio_urls,
        "missing":    missing,
    })


@app.route("/api/vocabulary")
def api_vocabulary():
    templates = load_templates()
    result = []
    for word, sym in VOCAB.items():
        count = len(templates.get(word, []))
        has_audio = os.path.exists(os.path.join(AUDIO_DIR, f"{word}.wav"))
        result.append({
            "word": word, "symbol": sym,
            "templates": count, "has_audio": has_audio
        })
    return jsonify(result)


@app.route("/api/reset/<word>", methods=["DELETE"])
def api_reset_word(word):
    word = word.lower()
    templates = load_templates()
    if word in templates:
        del templates[word]
        save_templates(templates)
    audio_path = os.path.join(AUDIO_DIR, f"{word}.wav")
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return jsonify({"success": True, "word": word})


if __name__ == "__main__":
    print("=" * 60)
    print("  ETS 2 — STT & TTS App")
    print("  NRP : 152023013")
    print("  Nama: Cleosa Zelda Avrillya")
    print("=" * 60)
    print("  Buka browser: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)