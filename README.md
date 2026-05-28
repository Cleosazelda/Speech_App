# 🎙 Aplikasi STT + TTS — ETS 2 IFB-306
**NRP:** 152023013  
**Nama:** Cleosa Zelda Avrillya  
**Metode:** MFCC (Mel-Frequency Cepstral Coefficients) + DTW (Dynamic Time Warping)

---

## 📦 Struktur Proyek

```
speech_app/
├── app.py                ← Backend Flask (MFCC + DTW engine)
├── requirements.txt      ← Dependensi Python
├── templates/
│   └── index.html        ← UI Web (perekaman, STT, TTS)
├── static/
│   └── audio/            ← File .wav hasil rekaman (untuk TTS)
├── recordings/           ← Folder backup rekaman
└── models/
    └── templates.pkl     ← Template MFCC tersimpan (untuk STT)
```

---

## 🚀 Step-by-Step Instalasi & Penggunaan

### STEP 1 — Instalasi Python & Dependensi

```bash
# Pastikan Python 3.8+ terinstal
python --version

# Install semua dependensi
pip install -r requirements.txt
```

Dependensi utama:
- `flask` — web server backend
- `librosa` — ekstraksi fitur MFCC dari audio
- `numpy` — komputasi matriks DTW
- `scipy` — distance calculations
- `soundfile` — baca/tulis file audio

---

### STEP 2 — Jalankan Aplikasi

```bash
python app.py
```

Output terminal:
```
============================================================
  ETS 2 — STT & TTS App
  NRP : 152023013
  Nama: Cleosa Zelda Avrillya
============================================================
  Buka browser: http://127.0.0.1:5000
============================================================
```

Buka browser dan akses: **http://127.0.0.1:5000**

---

### STEP 3 — Rekam Suara (Tab "Rekam Suara") 🎙

> ⚠️ **WAJIB menggunakan suara sendiri (Cleosa Zelda Avrillya)**

**Kata-kata yang harus direkam (18 kata total):**

| Kategori | Kata | Simbol |
|----------|------|--------|
| **Bilangan Asli** | nol, satu, dua, tiga, empat, lima, enam, tujuh, delapan, sembilan, sepuluh | 0–10 |
| **Operator Aritmatika** | tambah, kurang, kali, bagi | +, −, ×, ÷ |
| **Identitas (Nama)** | cleosa, zelda, avrillya | — |

> NRP `152023013` terdiri dari digit: satu, lima, dua, nol, dua, tiga, nol, satu, tiga  
> Semua digit sudah ada dalam kategori Bilangan Asli ✓

**Cara merekam:**
1. Klik tombol kata yang ingin direkam (misalnya "satu")
2. Tekan tombol 🎙 besar
3. Ucapkan kata tersebut dengan **jelas dan natural**
4. Tekan ⏹ untuk stop
5. Ulangi **minimal 3× per kata** untuk akurasi DTW lebih baik

**Tips rekaman:**
- Rekam di **tempat tenang** (minim noise)
- Ucapkan kata dengan **durasi normal** (~0.5–1 detik)
- Jaga **jarak konsisten** dari mikrofon (~15–20 cm)
- Ucapkan kata lengkap, bukan dieja per huruf

---

### STEP 4 — Uji Speech-to-Text (Tab "STT") 👂

**Cara pakai:**
1. Pastikan semua (atau sebagian) kata sudah direkam di Step 3
2. Buka tab **STT**
3. Tekan tombol 🎙
4. Ucapkan salah satu kata dari vocabulary
5. Tekan ⏹ stop
6. Lihat hasil pengenalan + jarak DTW + confidence

**Cara kerja di balik layar:**
```
Audio input
    ↓
Pre-processing (resample 16kHz, trim silence, normalize)
    ↓
MFCC Extraction (13 koefisien + delta + delta² = 39 fitur per frame)
    ↓
DTW Matching (bandingkan dengan semua template tersimpan)
    ↓
Nearest Neighbor (pilih kata dengan jarak DTW terkecil)
    ↓
Output: Kata + Symbol + Confidence
```

---

### STEP 5 — Uji Text-to-Speech (Tab "TTS") 🔊

**Cara pakai:**
1. Ketik teks di input field, contoh: `satu tambah dua`
2. Tekan tombol 🔊 atau Enter
3. Sistem akan memainkan rekaman suaramu satu per satu

**Contoh input yang didukung:**
| Input | Diucapkan |
|-------|-----------|
| `nrp` | satu-lima-dua-nol-dua-tiga-nol-satu-tiga |
| `nama` | cleosa-zelda-avrillya |
| `identitas` | [NRP] + [nama] |
| `satu tambah dua` | "satu" → "tambah" → "dua" |
| `sepuluh bagi dua` | "sepuluh" → "bagi" → "dua" |
| `nol satu dua tiga` | satu per satu |

---

### STEP 6 — Cek Status Dataset (Tab "Status") 📊

- Lihat berapa template yang sudah tersimpan per kata
- Cek apakah file audio TTS tersedia
- Gunakan tombol **Reset** jika ingin merekam ulang suatu kata

---

## 🔬 Penjelasan Teknis (untuk Laporan)

### Ekstraksi MFCC
```python
# Parameter yang digunakan:
n_mfcc     = 13      # Koefisien MFCC utama
n_fft      = 512     # FFT window size
hop_length = 160     # 10 ms @ 16kHz
win_length = 400     # 25 ms window (Hamming)
n_mels     = 26      # Mel filterbank

# Ditambah delta & delta² → 39 fitur total per frame
```

### Algoritma DTW
```python
# Pseudocode DTW dengan Sakoe-Chiba Band (radius=10):
D[0][0] = 0
for i in range(1, N+1):
    for j in range(max(1, i-R), min(M, i+R)+1):
        cost = euclidean(A[i], B[j])
        D[i][j] = cost + min(D[i-1][j], D[i][j-1], D[i-1][j-1])
return D[N][M] / (N+M)   # normalized distance
```

### Klasifikasi (Nearest Neighbor)
```python
predicted = argmin_word( min_over_templates( DTW(test, template) ) )
```

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| "Belum ada template" | Rekam data latih dulu di tab Rekam Suara |
| "Rekaman terlalu pendek" | Ucapkan kata lebih panjang, min ~0.3 detik |
| "Akses mikrofon ditolak" | Izinkan browser mengakses mikrofon |
| Audio TTS tidak bunyi | Pastikan kata sudah direkam (cek tab Status) |
| Akurasi STT rendah | Tambah jumlah template per kata (>5×) |

---

## 📋 Informasi Mata Kuliah

- **Mata Kuliah:** IFB-306 Pengenalan Ucapan & Teks ke Ucapan  
- **Dosen:** Asep Nana Hermana, S.T., M.T.  
- **ETS 2:** Semester Genap 2025/2026  
- **SubCPMK:** 4 & 5 (TTS Problem + Implementasi, Teks ke Fonem & Fonem ke Ucapan)
