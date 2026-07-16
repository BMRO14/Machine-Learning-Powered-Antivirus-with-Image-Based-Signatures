from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

IMAGE_ROOT   = Path(r"C:\Users\VMLicenta\Desktop\entropy")         
FEATURES_DIR = Path(r"C:\Users\VMLicenta\Desktop\features")     
MODEL_PATH   = Path(r"C:\Users\VMLicenta\Desktop\models\rf_train_plots.joblib")

PLOTS_DIR = Path("plots_multiclass")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ["ransomware", "trojan", "rat", "worm", "benign"]

STATIC_COLS = [
    "log_file_size",
    "window_entropy",
    "mean_chunk_entropy",
    "std_chunk_entropy",
    "max_chunk_entropy",
    "min_chunk_entropy",
    "high_entropy_ratio"
]

def load_static_features(features_dir: Path, class_names: list):

    lookup = {}

    for class_name in class_names:
        csv_path = features_dir / f"{class_name}_features.csv"

        if not csv_path.exists():
            print(f"Warning: CSV not found -> {csv_path}")
            lookup[class_name] = {}
            continue

        df = pd.read_csv(csv_path)

        class_lookup = {}
        for _, row in df.iterrows():
            stem = Path(row["file_name"]).stem   
            feat = row[STATIC_COLS].values.astype(np.float32)
            class_lookup[stem] = feat

        lookup[class_name] = class_lookup
        print(f"  Loaded {len(class_lookup)} static feature rows for '{class_name}'")

    return lookup

print("Loading static features from CSVs...")
static_lookup = load_static_features(FEATURES_DIR, CLASS_NAMES)

def load_combined_features(image_root: Path, class_names: list, static_lookup: dict):
    X = []
    y = []
    skipped = 0

    for label_idx, class_name in enumerate(class_names):
        class_dir = image_root / class_name

        if not class_dir.exists():
            print(f"Warning: image folder not found -> {class_dir}")
            continue

        class_static = static_lookup.get(class_name, {})

        for img_path in sorted(class_dir.glob("*.png")):
            stem = img_path.stem   
            static_feat = class_static.get(stem)
            if static_feat is None:
                skipped += 1
                continue   

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((8, 8))
                arr = np.array(img, dtype=np.float32)
                image_feat = arr.flatten()         
            except Exception as e:
                print(f"Skipping image {img_path}: {e}")
                skipped += 1
                continue

            combined = np.hstack([image_feat, static_feat])

            X.append(combined)
            y.append(label_idx)

    if skipped > 0:
        print(f"Warning: {skipped} images skipped (no matching static features or load error)")

    X = np.array(X)
    y = np.array(y)
    return X, y

print("\nLoading entropy images and merging features...")
X, y = load_combined_features(IMAGE_ROOT, CLASS_NAMES, static_lookup)

print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features each")
print(f"  Image features : 192")
print(f"  Static features:   7")
print(f"  Total           : {X.shape[1]}")

for idx, name in enumerate(CLASS_NAMES):
    print(f"  Class '{name}': {np.sum(y == idx)} samples")

if len(X) == 0:
    print("\nError: no samples loaded. Check that image filenames match CSV file_name stems.")
    exit()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"\nSplit sizes -> Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_and_save_normalized_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_and_save_feature_importance(model, feature_labels, save_path, top_n=15):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_labels[i] for i in idx]
    top_values = importances[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_values, color="steelblue")
    plt.yticks(range(len(top_features)), top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Most Important Features")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_and_save_class_metrics(y_true, y_pred, class_names, save_path, title):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    precision = [report[name]["precision"] for name in class_names]
    recall = [report[name]["recall"] for name in class_names]
    f1 = [report[name]["f1-score"] for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, class_names, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_split(name, model, X_split, y_split):
    y_pred = model.predict(X_split)

    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_split, y_pred), 4))
    print(classification_report(y_split, y_pred, target_names=CLASS_NAMES))
    print("Confusion matrix:")
    print(confusion_matrix(y_split, y_pred))

    plot_and_save_confusion_matrix(
        y_split,
        y_pred,
        CLASS_NAMES,
        title=f"{name} Confusion Matrix",
        save_path=PLOTS_DIR / f"{name.lower()}_confusion_matrix.png"
    )

    plot_and_save_normalized_confusion_matrix(
        y_split,
        y_pred,
        CLASS_NAMES,
        title=f"{name} Normalized Confusion Matrix",
        save_path=PLOTS_DIR / f"{name.lower()}_confusion_matrix_normalized.png"
    )

    plot_and_save_class_metrics(
        y_split,
        y_pred,
        CLASS_NAMES,
        save_path=PLOTS_DIR / f"{name.lower()}_class_metrics.png",
        title=f"{name} Per-Class Precision / Recall / F1"
    )

evaluate_split("Validation", rf, X_val, y_val)
evaluate_split("Test", rf, X_test, y_test)

feature_labels = [f"img_{i}" for i in range(192)] + STATIC_COLS
importances = rf.feature_importances_

print("\n--- Top 15 most important features ---")
top_idx = np.argsort(importances)[::-1][:15]
for rank, i in enumerate(top_idx, start=1):
    print(f"  {rank:2d}. {feature_labels[i]:30s}  importance: {importances[i]:.4f}")

plot_and_save_feature_importance(
    rf,
    feature_labels,
    save_path=PLOTS_DIR / "top_15_feature_importances.png",
    top_n=15
)

dump(
    {
        "model": rf,
        "class_names": CLASS_NAMES,
        "static_cols": STATIC_COLS,
        "feature_count": X.shape[1]
    },
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Plots saved to: {PLOTS_DIR.resolve()}")