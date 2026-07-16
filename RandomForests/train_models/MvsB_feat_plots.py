from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from joblib import dump

IMAGE_ROOT   = Path(r"C:\Users\VMLicenta\Desktop\entropy")          
FEATURES_DIR = Path(r"C:\Users\VMLicenta\Desktop\features")      
MODEL_PATH   = Path(r"C:\Users\VMLicenta\Desktop\models\MvsB_train_plots.joblib")

PLOTS_DIR = Path("plots_binary")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MALWARE_CLASSES = ["ransomware", "trojan", "rat", "worm"]
BENIGN_CLASS = "benign2"
CLASS_NAMES = ["malicious", "benign"]

STATIC_COLS = [
    "log_file_size",
    "window_entropy",
    "mean_chunk_entropy",
    "std_chunk_entropy",
    "max_chunk_entropy",
    "min_chunk_entropy",
    "high_entropy_ratio"
]

def load_static_features_binary(features_dir: Path):
    lookup = {}

    for class_name in MALWARE_CLASSES + [BENIGN_CLASS]:
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
static_lookup = load_static_features_binary(FEATURES_DIR)

def load_binary_combined_features(image_root: Path, static_lookup: dict):
    X = []
    y = []
    skipped = 0

    # malicious = 0
    for class_name in MALWARE_CLASSES:
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
            y.append(0)

    # benign = 1
    benign_dir = image_root / BENIGN_CLASS
    if not benign_dir.exists():
        print(f"Warning: image folder not found -> {benign_dir}")
    else:
        benign_static = static_lookup.get(BENIGN_CLASS, {})

        for img_path in sorted(benign_dir.glob("*.png")):
            stem = img_path.stem
            static_feat = benign_static.get(stem)

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
            y.append(1)

    if skipped > 0:
        print(f"Warning: {skipped} images skipped (no matching static features or load error)")

    X = np.array(X)
    y = np.array(y)
    return X, y

print("\nLoading entropy images and merging features...")
X, y = load_binary_combined_features(IMAGE_ROOT, static_lookup)

print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features each")
print(f"  Image features : 192")
print(f"  Static features:   7")
print(f"  Total           : {X.shape[1]}")
print(f"  Malicious       : {np.sum(y == 0)}")
print(f"  Benign          : {np.sum(y == 1)}")

if len(X) == 0:
    print("\nError: no samples loaded. Check that image filenames match CSV file_name stems.")
    exit()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train size:", X_train.shape[0])
print("Val size  :", X_val.shape[0])
print("Test size :", X_test.shape[0])


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
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

    plt.figure(figsize=(6, 5))
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

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, class_names)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_and_save_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=0)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_and_save_pr_curve(y_true, y_score, save_path):
    precision, recall, _ = precision_recall_curve(y_true == 0, y_score)
    ap = average_precision_score(y_true == 0, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}", color="purple")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_and_save_probability_histogram(y_true, y_score, save_path):
    plt.figure(figsize=(7, 5))
    plt.hist(y_score[y_true == 0], bins=20, alpha=0.6, label="Malicious")
    plt.hist(y_score[y_true == 1], bins=20, alpha=0.6, label="Benign")
    plt.xlabel("Predicted probability of malicious")
    plt.ylabel("Count")
    plt.title("Probability Distribution")
    plt.legend()
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

def evaluate_split(name, model, X_split, y_split):
    y_pred = model.predict(X_split)
    y_proba = model.predict_proba(X_split)[:, 0]   

    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_split, y_pred))
    print(classification_report(
        y_split,
        y_pred,
        target_names=CLASS_NAMES
    ))
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
        title=f"{name} Precision / Recall / F1"
    )

    plot_and_save_roc_curve(
        y_split,
        y_proba,
        save_path=PLOTS_DIR / f"{name.lower()}_roc_curve.png"
    )

    plot_and_save_pr_curve(
        y_split,
        y_proba,
        save_path=PLOTS_DIR / f"{name.lower()}_pr_curve.png"
    )

    plot_and_save_probability_histogram(
        y_split,
        y_proba,
        save_path=PLOTS_DIR / f"{name.lower()}_probability_histogram.png"
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