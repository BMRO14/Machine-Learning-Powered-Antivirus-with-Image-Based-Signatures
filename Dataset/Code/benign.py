import os
import shutil
import random

BASE_DIR = r"C:\Users\VMLicenta\Desktop\Dataset\malware_dataset\benign2"
BENIGN_DIR = os.path.join(BASE_DIR, "benign2")
TARGET_COUNT = 2000

BENIGN_SOURCES = [
    r"C:\Windows\System32",
    r"C:\Windows\SysWOW64",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
]

MIN_SIZE = 20_000       
MAX_SIZE = 50_000_000   

os.makedirs(BENIGN_DIR, exist_ok=True)

def collect_benign(out_dir, target=100):
    collected = 0
    candidates = []

    print("[*] Scanning for benign .exe files...")
    for src_dir in BENIGN_SOURCES:
        if not os.path.exists(src_dir):
            continue
        for root, _, files in os.walk(src_dir):
            for fname in files:
                if not fname.lower().endswith(".exe"):
                    continue
                full_path = os.path.join(root, fname)
                try:
                    size = os.path.getsize(full_path)
                    if MIN_SIZE < size < MAX_SIZE:
                        candidates.append(full_path)
                except OSError:
                    continue

    print(f"[*] Found {len(candidates)} candidate benign executables")

    if not candidates:
        print("[!] No candidates found. Check BENIGN_SOURCES paths.")
        return 0

    random.shuffle(candidates)
    selected = candidates[:target]

    for src_path in selected:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(out_dir, fname)

        # Avoid name collisions
        base, ext = os.path.splitext(fname)
        i = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(out_dir, f"{base}_{i}{ext}")
            i += 1

        try:
            shutil.copy2(src_path, dst_path)
            collected += 1
            print(f"[{collected}/{target}] {fname}")
        except Exception as e:
            print(f"[!] Failed to copy {fname}: {e}")

        if collected >= target:
            break

    print(f"[*] Done. Collected {collected} benign files into: {out_dir}")
    return collected

if __name__ == "__main__":
    collect_benign(BENIGN_DIR, TARGET_COUNT)