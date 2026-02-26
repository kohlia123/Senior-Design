import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay
)

from src.utils.preprocessing import get_subj_data
from src.config import N_SUB

FIXED_THRESHOLD = 0.85
MODEL_PATH = "models/trained_model.joblib"  # update this to wherever you save your model
SUBJECT_ID = "01"  # change this to whichever subject you want to test

def test_single_subject(subject_id, model_path, threshold=FIXED_THRESHOLD):
    print(f"--- Testing on Subject {subject_id} ---")

    # 1. Load subject data
    X_raw, y = get_subj_data(subject_id)
    y = np.asarray(y, dtype=int)

    # 2. Drop metadata columns
    feature_cols = [c for c in X_raw.columns if c not in ['subj', 'chan_name', 'epoch', 'epoch_id']]
    X = X_raw[feature_cols]

    # 3. Load trained model
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

    # 4. Predict using fixed threshold
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    print(f"Using classification threshold: {threshold}")

    # 5. Metrics
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy":    accuracy_score(y, y_pred),
        "precision":   precision_score(y, y_pred, zero_division=0),
        "sensitivity": recall_score(y, y_pred, zero_division=0),
        "specificity": specificity,
        "f1":          f1_score(y, y_pred, zero_division=0),
        "ROCAUC":      roc_auc_score(y, y_score) if len(np.unique(y)) == 2 else None,
        "PRAUC":       average_precision_score(y, y_score) if len(np.unique(y)) == 2 else None,
    }

    print("\nMetrics Table:")
    metrics_df = pd.DataFrame([metrics])
    print(metrics_df.to_string(index=False))

    # 6. Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Spike", "Spike"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Confusion Matrix — Subject {subject_id}")
    plt.tight_layout()
    plt.show()

    return metrics_df

if __name__ == "__main__":
    test_single_subject(SUBJECT_ID, MODEL_PATH)