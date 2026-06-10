import os
import sys
import pickle
import soundfile as sf
import numpy as np

# Add local directory to path to import app helpers
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app import load_audio_file, preprocess_audio, extract_mfcc, save_templates, VOCAB, AUDIO_DIR

def get_base_name(filename):
    base = filename.lower()
    for ext in [".m4a.mp4", ".mp4", ".m4a", ".wav"]:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    return base.strip()

def process_file(file_path, word, target_dir, templates):
    try:
        # Load audio using FFmpeg helper (handles m4a, wav, mp4, etc.)
        y, sr = load_audio_file(file_path)
        
        # Preprocess: resample, mono, normalize, trim
        y_proc, sr_proc = preprocess_audio(y, sr)
        
        # Extract MFCC features
        mfcc = extract_mfcc(y_proc, sr_proc)
        
        # Add to templates dictionary
        if word not in templates:
            templates[word] = []
        templates[word].append(mfcc)
        
        # Target wav path inside the category folder
        dest_path = os.path.join(target_dir, f"{word}.wav")
        sf.write(dest_path, y_proc, sr_proc)
        print(f"    [OK] {word} -> {dest_path} (Frames: {len(mfcc)})")
        
        # If the original file was not wav, or has different casing/path, delete the original file!
        # Note: comparison must be case-sensitive and path-sensitive to prevent self-deletion
        if not file_path.lower().endswith(".wav") or os.path.abspath(file_path) != os.path.abspath(dest_path):
            try:
                os.remove(file_path)
                print(f"    [CLEAN] Deleted original non-wav/uncased file: {file_path}")
            except Exception:
                pass
    except Exception as e:
        print(f"    [ERR] Gagal memproses {word} ({file_path}): {e}")

def main():
    print("============================================================")
    print("  UAS - Speech App Dataset Training, Cleanup & Sync Script")
    print("============================================================")
    templates = {}
    
    # 1. Alphabet
    alphabet_dir = os.path.join(AUDIO_DIR, "Alphabet", "Aphabet")
    if os.path.exists(alphabet_dir):
        print(f"\n[+] Scanning & cleaning Alphabet folder: {alphabet_dir}")
        for letter in "abcdefghijklmnopqrstuvwxyz":
            # Possible file names: A.m4a, a.m4a, A.wav, a.wav, etc.
            found_file = None
            for ext in [".wav", ".m4a", ".mp4"]:
                for casing in [letter.upper(), letter.lower()]:
                    path = os.path.join(alphabet_dir, f"{casing}{ext}")
                    if os.path.exists(path):
                        found_file = path
                        break
                if found_file:
                    break
            
            if found_file:
                process_file(found_file, letter, alphabet_dir, templates)
            else:
                print(f"[-] Warning: Letter file for '{letter}' not found.")
    else:
        print(f"[-] Alphabet directory not found at: {alphabet_dir}")

    # 2. Angka
    angka_dir = os.path.join(AUDIO_DIR, "Angka")
    if os.path.exists(angka_dir):
        print(f"\n[+] Scanning & cleaning Angka folder: {angka_dir}")
        for file in os.listdir(angka_dir):
            file_path = os.path.join(angka_dir, file)
            if os.path.isdir(file_path):
                continue
            base = get_base_name(file)
            if base in VOCAB:
                process_file(file_path, base, angka_dir, templates)
    else:
        print(f"[-] Angka directory not found at: {angka_dir}")

    # 3. Aritmatika
    aritmatika_dir = os.path.join(AUDIO_DIR, "Aritmatika")
    if os.path.exists(aritmatika_dir):
        print(f"\n[+] Scanning & cleaning Aritmatika folder: {aritmatika_dir}")
        for file in os.listdir(aritmatika_dir):
            file_path = os.path.join(aritmatika_dir, file)
            if os.path.isdir(file_path):
                continue
            base = get_base_name(file)
            if base in VOCAB:
                process_file(file_path, base, aritmatika_dir, templates)
    else:
        print(f"[-] Aritmatika directory not found at: {aritmatika_dir}")

    # 4. Frasa
    frasa_dir = os.path.join(AUDIO_DIR, "Frasa")
    if os.path.exists(frasa_dir):
        print(f"\n[+] Scanning & cleaning Frasa folder: {frasa_dir}")
        for file in os.listdir(frasa_dir):
            file_path = os.path.join(frasa_dir, file)
            if os.path.isdir(file_path):
                continue
            base = get_base_name(file)
            if base in VOCAB:
                process_file(file_path, base, frasa_dir, templates)
            else:
                # Keep it if it's a valid wav, or delete if not wav
                if not file.lower().endswith(".wav"):
                    try:
                        os.remove(file_path)
                        print(f"    [CLEAN] Deleted non-vocab non-wav: {file}")
                    except Exception:
                        pass

    # Clean up non-wav files in all subfolders
    print("\n[+] Finalizing folder cleanliness (deleting any remaining non-wav files)...")
    for root, dirs, files in os.walk(AUDIO_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(".wav"):
                try:
                    os.remove(file_path)
                    print(f"    [CLEAN] Deleted non-wav file: {file_path}")
                except Exception:
                    pass

    # Clean up any wav files in the root of AUDIO_DIR (static/audio/)
    print("\n[+] Finalizing root cleanliness (deleting scattered wav files in root static/audio)...")
    for file in os.listdir(AUDIO_DIR):
        file_path = os.path.join(AUDIO_DIR, file)
        if os.path.isfile(file_path) and file.lower().endswith(".wav"):
            try:
                os.remove(file_path)
                print(f"    [CLEAN] Deleted root wav file: {file_path}")
            except Exception:
                pass

    # Save templates
    save_templates(templates)
    print("\n============================================================")
    print(f"  Training & Cleanup Selesai! Berhasil melatih {len(templates)} kosakata.")
    print("  Semua file selain .wav berhasil dihapus.")
    print("  Model templates.pkl berhasil disimpan.")
    print("============================================================")

if __name__ == "__main__":
    main()
