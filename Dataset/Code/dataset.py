import requests
import os
import time
import subprocess
import tempfile
import shutil

API_URL = "https://mb-api.abuse.ch/api/v1/"
ZIP_PASSWORD = "infected"
BASE_DIR = "malware_dataset"
SAMPLES_PER_CATEGORY = 500      
PER_TAG_LIMIT = 100            
SEVENZIP = r"C:\Program Files\7-Zip\7z.exe" 

AUTH_KEY = "API_KEY_HERE"
HEADERS = {"Auth-Key": AUTH_KEY}

CATEGORIES = {
    "ransomware": [
        "ransomware",
        "lockbit", "LockBit", "Ransomware.LockBit",
        "wannacry", "WannaCry", "Ransomware.WannaCry",
        "gandcrab", "GandCrab",
        "cerber", "Cerber",
        "dharma", "Dharma",
        "maze", "Maze",
        "phobos", "Phobos",
        "ryuk", "Ryuk",
        "conti", "Conti",
        "sodinokibi", "Sodinokibi", "revil", "REvil",
        "clop", "Clop",
        "darkside", "DarkSide",
        "ragnarlocker", "RagnarLocker",
        "blackmatter", "BlackMatter",
        "babuk", "Babuk",
        "zeppelin", "Zeppelin",
    ],

    "trojan": [
        "trojan",
        "emotet", "Emotet",
        "qbot", "Qbot", "qakbot", "QakBot",
        "dridex", "Dridex",
        "trickbot", "TrickBot",
        "ursnif", "Ursnif",
        "gozi", "Gozi",
        "icedid", "IcedID", "bokbot",
        "zeus", "Zeus",
        "agenttesla", "AgentTesla",
        "formbook", "Formbook",
        "azorult", "Azorult",
        "lokibot", "LokiBot",
        "vidar", "Vidar",
        "redline", "RedLine", "RedLineStealer",
        "raccoon", "RaccoonStealer",
    ],

    "rat": [
        "rat",
        "remcos", "Remcos", "RemcosRAT",
        "nanocore", "NanoCore",
        "njrat", "Njrat", "njRAT",
        "asyncrat", "AsyncRAT",
        "quasar", "QuasarRAT", "Quasar",
        "netwire", "NetWire",
        "darkcomet", "DarkComet",
        "gh0st", "Gh0st", "Gh0stRAT",
        "plugx", "PlugX",
        "adwind", "Adwind",
        "poisonivy", "PoisonIvy",
        "xrat", "Xrat",
    ],

    "worm": [
        "worm",
        "conficker", "Conficker",
        "sality", "Sality",
        "gamarue", "Gamarue",
        "autorun", "INF/Autorun",
        "virut", "Virut",
        "vobfus", "Vobfus",
        "dorkbot", "Dorkbot",
        "ramnit", "Ramnit",
        "nimda", "Nimda",
        "kido", "Kido",
        "blaster", "Blaster",
        "sasser", "Sasser",
    ],
}

def query_by_tag(tag, limit=PER_TAG_LIMIT):
    try:
        r = requests.post(
            API_URL,
            headers=HEADERS,
            data={"query": "get_taginfo", "tag": tag, "limit": limit},
            timeout=15,
        )
        data = r.json()
        print(f"    API status: {data.get('query_status')}")
        if data.get("query_status") == "ok":
            return [
                s["sha256_hash"]
                for s in data["data"]
                if s.get("file_type") in ("exe", "dll", "pe32", "pe32+")
            ]
    except Exception as e:
        print(f"    [!] Query error for tag '{tag}': {e}")
    return []

def download_sample(sha256, out_dir):
    try:
        r = requests.post(
            API_URL,
            headers=HEADERS,
            data={"query": "get_file", "sha256_hash": sha256},
            timeout=30,
        )
        if r.status_code != 200 or r.content[:2] != b"PK":
            print(f"    [!] Bad response for {sha256[:16]}")
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "sample.zip")
            with open(zip_path, "wb") as f:
                f.write(r.content)

            result = subprocess.run(
                [SEVENZIP, "e", zip_path, f"-p{ZIP_PASSWORD}", f"-o{tmpdir}", "-y"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"    [!] 7z failed for {sha256[:16]}: {result.stderr.strip()}")
                return False

            for fname in os.listdir(tmpdir):
                if fname == "sample.zip":
                    continue
                src = os.path.join(tmpdir, fname)
                dst = os.path.join(out_dir, sha256[:16] + ".exe")
                shutil.move(src, dst)
                return True

    except Exception as e:
        print(f"    [!] Error for {sha256[:16]}: {e}")
    return False

for category, tags in CATEGORIES.items():
    out_dir = os.path.join(BASE_DIR, category)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Category: {category.upper()}")
    print(f"{'='*60}")

    collected = 0
    seen = set()

    for tag in tags:
        if collected >= SAMPLES_PER_CATEGORY:
            break

        print(f"  Querying tag: '{tag}'...")
        hashes = query_by_tag(tag, limit=PER_TAG_LIMIT)
        print(f"  → {len(hashes)} PE samples found")

        for h in hashes:
            if collected >= SAMPLES_PER_CATEGORY:
                break
            if h in seen:
                continue
            seen.add(h)

            if download_sample(h, out_dir):
                collected += 1
                print(f"  [{collected}/{SAMPLES_PER_CATEGORY}] {h[:20]}...")
            time.sleep(0.7) 

    print(f"  Done: {collected} files in '{out_dir}'")

print("\nComplete. Dataset structure:")
for cat in CATEGORIES:
    p = os.path.join(BASE_DIR, cat)
    n = len(os.listdir(p)) if os.path.exists(p) else 0
    print(f"  {p}/  ({n} files)")
