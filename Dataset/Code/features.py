from pathlib import Path
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

WINDOW_SIZE = 16 * 1024
CHUNK_SIZE = 256
NUM_CHUNKS = WINDOW_SIZE // CHUNK_SIZE
HIGH_ENTROPY_THRESHOLD = 7.0

OUTPUT_DIR = Path(r"C:\Users\VMLicenta\Desktop\features\benign2")

def shannon_entropy(byte_data: bytes) -> float:
    if not byte_data:
        return 0.0
    arr = np.frombuffer(byte_data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / len(arr)
    return float(-np.sum(probs * np.log2(probs)))

def extract_7_features(file_path: Path):
    try:
        raw = file_path.read_bytes()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return None

    if len(raw) == 0:
        print(f"Skipping empty file: {file_path}")
        return None

    file_size = len(raw)
    window = raw[:WINDOW_SIZE]

    if len(window) < WINDOW_SIZE:
        window = window + bytes(WINDOW_SIZE - len(window))

    chunk_entropies = []
    for i in range(NUM_CHUNKS):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = window[start:end]
        ent = shannon_entropy(chunk)
        chunk_entropies.append(ent)

    chunk_entropies = np.array(chunk_entropies, dtype=np.float32)

    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_size_bytes": file_size,
        "log_file_size": float(np.log1p(file_size)),
        "window_entropy": float(shannon_entropy(window)),
        "mean_chunk_entropy": float(np.mean(chunk_entropies)),
        "std_chunk_entropy": float(np.std(chunk_entropies)),
        "max_chunk_entropy": float(np.max(chunk_entropies)),
        "min_chunk_entropy": float(np.min(chunk_entropies)),
        "high_entropy_ratio": float(np.mean(chunk_entropies > HIGH_ENTROPY_THRESHOLD))
    }

FIELDNAMES = [
    "file_name",
    "file_path",
    "file_size_bytes",
    "log_file_size",
    "window_entropy",
    "mean_chunk_entropy",
    "std_chunk_entropy",
    "max_chunk_entropy",
    "min_chunk_entropy",
    "high_entropy_ratio"
]

def save_csv(category: str, rows: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{category}_features.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved: {csv_path}  ({len(rows)} rows)")

def main():
    root = tk.Tk()
    root.withdraw()

    selected_files = filedialog.askopenfilenames(
        title="Select executable files to process",
        filetypes=[
            ("Executable files", "*.exe *.dll *.bin"),
            ("All files", "*.*")
        ]
    )

    if not selected_files:
        print("No files selected.")
        return

    print(f"\nSelected {len(selected_files)} file(s).\n")
    grouped = defaultdict(list)
    for filepath_str in selected_files:
        fp = Path(filepath_str)
        category = fp.parent.name  
        grouped[category].append(fp)

    total_processed = 0

    for category, files in grouped.items():
        print(f"Processing category: '{category}'  ({len(files)} files)")
        rows = []

        for idx, fp in enumerate(files, start=1):
            result = extract_7_features(fp)
            if result is not None:
                rows.append(result)
            print(f"  [{idx}/{len(files)}] {fp.name}")

        if rows:
            save_csv(category, rows, OUTPUT_DIR)
            total_processed += len(rows)
        else:
            print(f"  No valid files processed for '{category}'")

    print(f"\nDone. {total_processed} files processed in total.")
    print(f"CSVs saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()