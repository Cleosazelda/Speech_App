import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app import load_audio_file, preprocess_audio, extract_mfcc, load_templates, dtw_distance

def test_recognition(file_path, expected_label):
    print(f"[TEST] File: {file_path}")
    if not os.path.exists(file_path):
        print(f"  [ERR] File {file_path} not found!")
        return False
        
    try:
        y, sr = load_audio_file(file_path)
        y, sr = preprocess_audio(y, sr)
        mfcc = extract_mfcc(y, sr)
        
        templates = load_templates()
        if not templates:
            print("  [ERR] Templates store is empty!")
            return False
            
        best_word = None
        best_dist = float("inf")
        
        for word, mfcc_list in templates.items():
            word_dists = [dtw_distance(mfcc, np.array(tmpl)) for tmpl in mfcc_list]
            min_dist = min(word_dists)
            if min_dist < best_dist:
                best_dist = min_dist
                best_word = word
                
        print(f"  -> Terdeteksi: '{best_word}' (Ekspektasi: '{expected_label}') · Jarak DTW: {best_dist:.4f}")
        if best_word == expected_label:
            print("  -> [PASSED]")
            return True
        else:
            print(f"  -> [FAILED] Cocok dengan '{best_word}'")
            return False
    except Exception as e:
        print(f"  -> [ERR] Terjadi kesalahan: {e}")
        return False

def main():
    print("============================================================")
    print("  UAS - Verifikasi Jalur Pengenalan Kasus Uji (DTW)")
    print("============================================================")
    results = []
    
    # Uji Kasus Alphabet (semua .wav lowercase)
    results.append(test_recognition("static/audio/Alphabet/Aphabet/a.wav", "a"))
    results.append(test_recognition("static/audio/Alphabet/Aphabet/b.wav", "b"))
    
    # Uji Kasus Angka
    results.append(test_recognition("static/audio/Angka/satu.wav", "satu"))
    results.append(test_recognition("static/audio/Angka/dua.wav", "dua"))
    
    # Uji Kasus Aritmatika
    results.append(test_recognition("static/audio/Aritmatika/tambah.wav", "tambah"))
    
    # Uji Kasus Frasa Utuh UAS (semua .wav lowercase)
    results.append(test_recognition("static/audio/Frasa/1+1=2.wav", "1+1=2"))
    results.append(test_recognition("static/audio/Frasa/123sayangsemuanya.wav", "123sayangsemuanya"))
    
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print("\n============================================================")
    print(f"  Hasil Pengujian: {success_count} / {total_count} Kasus Lolos.")
    if success_count == total_count:
        print("  Semua jalur pengenalan DTW terverifikasi sukses!")
    else:
        print("  Ada beberapa pengujian yang gagal/tidak cocok.")
    print("============================================================")

if __name__ == "__main__":
    main()
