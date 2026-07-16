import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

CHUNK_SIZE = 256
NUM_CHUNKS = 64
TOTAL_BYTES = CHUNK_SIZE * NUM_CHUNKS  
IMAGE_SIZE = 8        
DISPLAY_SIZE = 512    

OUTPUT_DIR = r"C:\Users\VMLicenta\Desktop\entropy\benign2"

def compute_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = [0] * 256
    for byte in data:
        freq[byte] += 1
    entropy = 0.0
    length = len(data)
    for count in freq:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    return entropy


def entropy_to_rgb(normalized: float) -> tuple:
    v = max(0.0, min(1.0, normalized))
    if v <= 0.5:
        r = int(v * 2 * 255)
        g = 255
    else:
        r = 255
        g = int((1.0 - v) * 2 * 255)
    return (r, g, 0)


def process_file(filepath: str) -> str:
    with open(filepath, "rb") as f:
        raw = f.read(TOTAL_BYTES)

    if len(raw) < TOTAL_BYTES:
        raw = raw + b"\x00" * (TOTAL_BYTES - len(raw))

    pixels = []
    for i in range(NUM_CHUNKS):
        chunk = raw[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        entropy = compute_entropy(chunk)
        normalized = entropy / 8.0
        pixels.append(entropy_to_rgb(normalized))

    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    img.putdata(pixels)
    img_scaled = img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(OUTPUT_DIR, base_name + ".png")
    img_scaled.save(output_path)
    return output_path


def main():
    root = tk.Tk()
    root.withdraw()

    filepaths = filedialog.askopenfilenames(
        title="Select executable files",
        filetypes=[
            ("Executable files", "*.exe *.dll *.sys *.scr *.com"),
            ("All files", "*.*"),
        ],
    )

    if not filepaths:
        messagebox.showinfo("Cancelled", "No file selected.")
        return

    outputs = []
    errors = []

    for fp in filepaths:
        try:
            out = process_file(fp)
            outputs.append(out)
            print(f"[OK] Saved: {out}")
        except Exception as e:
            errors.append((fp, str(e)))
            print(f"[ERROR] {fp}: {e}")

    if outputs:
        messagebox.showinfo(
            "Done",
            "Entropy color matrices saved:\n" + "\n".join(outputs),
        )
    if errors:
        msg = "Some files failed:\n" + "\n".join(f"{p}: {err}" for p, err in errors)
        messagebox.showwarning("Errors", msg)


if __name__ == "__main__":
    main()