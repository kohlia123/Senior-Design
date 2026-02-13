# train.py
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

from lightgbm import LGBMClassifier

# adjust import path to wherever your preprocessing.py lives
from src.utils.preprocessing import get_subj_data
from src.config import N_SUB


def build_dataset(subjects):
    X_all = []
    y_all = []

    for subj in subjects:
        print(f"Loading sub-{subj} ...")
        X_subj, y_subj = get_subj_data(subj)

        # Drop non-numeric / debug columns that can't go into the model
        if "epoch" in X_subj.columns:
            X_subj = X_subj.drop(columns=["epoch"])

        X_all.append(X_subj)
        y_all.extend(y_subj)

    X = pd.concat(X_all, axis=0, ignore_index=True)
    y = np.asarray(y_all, dtype=int)

    # LightGBM can use pandas categorical features
    for c in ["subj", "chan_name"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return X, y


def run_kfold(X, y, n_splits=5, random_state=15):
    metrics = {
        "accuracy": [], "precision": [], "sensitivity": [], "specificity": [],
        "f1": [], "ROCAUC": [], "PRAUC": []
    }

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(X, y):
        model = LGBMClassifier(random_state=random_state)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["sensitivity"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        metrics["specificity"].append(tn / (tn + fp) if (tn + fp) else 0.0)

        # AUCs (safe in case a fold somehow has 1 class)
        if len(np.unique(y_test)) == 2:
            metrics["ROCAUC"].append(roc_auc_score(y_test, y_score))
            metrics["PRAUC"].append(average_precision_score(y_test, y_score))
        else:
            metrics["ROCAUC"].append(np.nan)
            metrics["PRAUC"].append(np.nan)

    results = pd.DataFrame(metrics)
    results.loc["mean"] = results.mean(numeric_only=True)
    return results


if __name__ == "__main__":
    subjects = [f"{i:02d}" for i in range(1, N_SUB + 1)]
    X_feat, y = build_dataset(subjects)

    results = run_kfold(X_feat, y, n_splits=5, random_state=15)
    print(results)