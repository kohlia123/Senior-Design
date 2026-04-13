import os
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve
)

# Internal imports: 
from src.utils.preprocessing import get_subj_data
from src.config import N_SUB


# Display settings:
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 6)

# Constants: 
RANDOM_STATE = 15

# Metadata columns that should never be passed to the model as features
META_COLS = ['subj', 'chan_name', 'epoch', 'epoch_id', 'event_id', 'onset_time']


# Dataset construction:

def build_dataset(subjects: list) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and combine feature data across all subjects in parallel.

    Each call to get_subj_data returns:
      - res[0]: DataFrame of extracted features for one subject
      - res[1]: array of labels (0 = non-IED, 1 = IED)

    Returns
    -------
    X : pd.DataFrame
        Combined feature matrix across all subjects.
    y : np.ndarray
        Combined label array.
    """
    results = Parallel(n_jobs=-1)(delayed(get_subj_data)(s) for s in subjects)
    # Sequential fallback for debugging (uncomment if parallel errors are unclear):
    # results = [get_subj_data(s) for s in subjects]

    X_all = [res[0] for res in results]
    y_all = [res[1] for res in results]

    X = pd.concat(X_all, axis=0, ignore_index=True)
    y = np.concatenate(y_all).astype(int)

    # Sanity check: feature and label counts must match
    assert len(X) == len(y), "Feature/label length mismatch after build_dataset"

    # Report class balance so imbalance is visible before training
    print(f"Dataset built: {len(X)} samples total, "
          f"{y.sum()} IED ({100 * y.mean():.1f}%), "
          f"{(y == 0).sum()} non-IED ({100 * (1 - y.mean()):.1f}%)")

    return X, y


# Training pipeline: 

def run_training_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Train and evaluate a LightGBM IED classifier using subject-grouped
    stratified k-fold cross-validation.

    Design choices:
    - StratifiedGroupKFold: keeps subjects isolated across folds (no data
      leakage between patients) while preserving the IED/non-IED ratio.
    - RandomUnderSampler: balances the training set by discarding excess
      non-IED windows before fitting.
    - Threshold tuning: the decision threshold is chosen on a held-out
      validation split (not the test fold) to avoid optimistic thresholds.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix including metadata columns.
    y : np.ndarray
        Binary label array (0 = non-IED, 1 = IED).
    n_splits : int
        Number of cross-validation folds.

    Returns
    -------
    results : pd.DataFrame
        Per-fold and mean metrics table.
    """

    # Derive feature columns by excluding known metadata
    feature_cols = [c for c in X.columns if c not in META_COLS]

    # Guard: ensure no non-numeric columns slipped into the feature set
    assert X[feature_cols].select_dtypes(exclude='number').empty, \
        "Non-numeric columns detected in feature set — check META_COLS"

    # Subject labels used to keep patients isolated across folds
    groups = X['subj']

    # Cross-validation splitter: stratified by label, grouped by subject
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Undersampler: randomly removes majority-class (non-IED) samples to balance training
    rus = RandomUnderSampler(random_state=RANDOM_STATE)

    # Storage for per-fold metrics and SHAP values
    metrics = {
        "accuracy": [],"precision": [], "sensitivity": [], "specificity": [],
        "f1": [], "ROCAUC": [], "PRAUC": [], "train_time_s": []
    }
    all_shap_values = []
    all_test_features = []
    best_model = None
    best_f1 = -1.0

    # Cross validation loop:
    for fold, (train_index, test_index) in enumerate(kf.split(X, y, groups=groups)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1} / {n_splits}")
        print(f"{'='*50}")

        try:
            # ── 1. Split into train and test for this fold ────────────────────
            X_train_full = X.iloc[train_index][feature_cols]
            X_test       = X.iloc[test_index][feature_cols]
            y_train_full = y[train_index]
            y_test       = y[test_index]

            print(f"  Train: {len(X_train_full)} samples "
                  f"({y_train_full.sum()} IED, {(y_train_full==0).sum()} non-IED)")
            print(f"  Test:  {len(X_test)} samples "
                  f"({y_test.sum()} IED, {(y_test==0).sum()} non-IED)")

            # ── 2. Carve out a validation set for threshold tuning ────────────
            # The validation set is held out from training and used only to
            # find the best decision threshold — it is NOT the test fold.
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.2,
                stratify=y_train_full,
                random_state=RANDOM_STATE
            )

            # ── 3. Undersample training set to balance classes ────────────────
            # Only the training split (not validation or test) is resampled.
            X_tr_res, y_tr_res = rus.fit_resample(X_tr, y_tr)

            # ── 4. Impute missing features ────────────────────────────────────
            # NaNs arise when a feature cannot be computed (e.g. window too
            # short). We add a binary indicator column per affected feature
            # so the model can distinguish "truly zero" from "not computed",
            # then fill the NaN itself with 0.
            missing_indicator_cols = []
            for col in feature_cols:
                if X_tr_res[col].isna().any() or X_val[col].isna().any():
                    ind_col = f"{col}_missing"
                    X_tr_res = X_tr_res.copy()
                    X_val    = X_val.copy()
                    X_test   = X_test.copy()
                    X_tr_res[ind_col] = X_tr_res[col].isna().astype(int)
                    X_val[ind_col]    = X_val[col].isna().astype(int)
                    X_test[ind_col]   = X_test[col].isna().astype(int)
                    missing_indicator_cols.append(ind_col)

            X_tr_res = X_tr_res.fillna(0)
            X_val    = X_val.fillna(0)
            X_test   = X_test.fillna(0)

            # ── 5. Train model ────────────────────────────────────────────────
            model = LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,   # prevents splits on very small leaf groups
                reg_alpha=0.1,          # L1 regularisation — penalises unused features
                reg_lambda=0.1,         # L2 regularisation — reduces overfitting
                random_state=RANDOM_STATE,
                verbose=-1              # suppress per-iteration LightGBM output
            )

            start = time.time()
            model.fit(X_tr_res, y_tr_res)
            train_time = time.time() - start
            print(f"  Training time: {train_time:.2f}s")

            # ── 6. Tune decision threshold on validation set ──────────────────
            # We sweep thresholds on the validation set (not the test fold) and
            # pick the one that maximises F1. This prevents the threshold from
            # being tuned to test-set patterns.
            val_scores = model.predict_proba(X_val)[:, 1]
            prec_curve, rec_curve, thresholds = precision_recall_curve(y_val, val_scores)
            f1_curve = (2 * prec_curve * rec_curve /
                        (prec_curve + rec_curve + 1e-9))
            best_thresh = thresholds[np.argmax(f1_curve[:-1])]
            print(f"  Best threshold (from validation set): {best_thresh:.3f}")

            # ── 7. Evaluate on held-out test fold ─────────────────────────────
            y_score = model.predict_proba(X_test)[:, 1]
            y_pred  = (y_score >= best_thresh).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
            print(f"  Precision={tp/(tp+fp+1e-9):.3f}  "
                  f"Sensitivity={tp/(tp+fn+1e-9):.3f}  "
                  f"Specificity={tn/(tn+fp+1e-9):.3f}")

            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            metrics["sensitivity"].append(recall_score(y_test, y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            metrics["specificity"].append(tn / (tn + fp) if (tn + fp) else 0.0)
            metrics["train_time_s"].append(train_time)

            if len(np.unique(y_test)) == 2:
                metrics["ROCAUC"].append(roc_auc_score(y_test, y_score))
                metrics["PRAUC"].append(average_precision_score(y_test, y_score))
            else:
                # Only one class present in this test fold — AUC undefined
                metrics["ROCAUC"].append(np.nan)
                metrics["PRAUC"].append(np.nan)

            # Track best model across folds by F1
            fold_f1 = metrics["f1"][-1]
            if fold_f1 > best_f1:
                best_f1 = fold_f1
                best_model = model
                best_thresh_final = best_thresh
                print(f"  *** New best model (F1={best_f1:.3f}) ***")

            # ── 8. Confusion matrix plot ──────────────────────────────────────
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                        xticklabels=['Normal', 'IED'],
                        yticklabels=['Normal', 'IED'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title(f'Confusion Matrix — Fold {fold + 1}')
            plt.tight_layout()
            plt.show()

            # ── 9. Collect SHAP values ────────────────────────────────────────
            # Sample from test fold in a stratified way so both classes are
            # represented in the SHAP summary, then compute tree SHAP values.
            n_sample = min(len(X_test), 100)
            samples_per_class = n_sample // 2
            pos_idx = X_test.index[y_test == 1]
            neg_idx = X_test.index[y_test == 0]
            sampled_idx = np.concatenate([
                np.random.choice(pos_idx, min(len(pos_idx), samples_per_class), replace=False),
                np.random.choice(neg_idx, min(len(neg_idx), samples_per_class), replace=False)
            ])
            X_test_sample = X_test.loc[sampled_idx]

            explainer = shap.TreeExplainer(model)
            shap_v = np.array(explainer.shap_values(X_test_sample))

            # For binary LightGBM, shap_values returns shape (2, n, f) — take class 1
            if shap_v.ndim == 3:
                shap_v = shap_v[1]

            all_shap_values.append(shap_v)
            all_test_features.append(X_test_sample)

        except Exception as e:
            print(f"  Fold {fold + 1} FAILED: {e}")
            raise

    # Results summary:
    results = pd.DataFrame(metrics)
    results.loc["mean"] = results.mean(numeric_only=True)
    print("\nFinal Results:")
    print(results)

    # SHAP summary plot:
    if all_test_features:
        final_shap     = np.vstack(all_shap_values)
        final_features = pd.concat(all_test_features, ignore_index=True)
        # Align feature columns in case indicator columns vary across folds
        final_features = final_features.reindex(columns=feature_cols, fill_value=0)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(final_shap, final_features, plot_type='bar', color='#b0acf7')
        plt.tight_layout()
        plt.show()
    else:
        print("No SHAP values collected — check fold errors above.")

    # Save best model:
    # Saves the fold with the highest F1, not just the last fold.
    # The decision threshold is saved alongside the model since it is
    # data-derived and must match the model at inference time.
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": best_model, "threshold": best_thresh_final},
        "models/trained_model.joblib"
    )
    print(f"\nBest model (F1={best_f1:.3f}) saved to models/trained_model.joblib")

    return results


# Call the training pipeline:

if __name__ == "__main__":
    all_subjects = [f"{i:02d}" for i in range(1, N_SUB + 1)]
    subjects = [s for s in all_subjects if 'openieeg' not in s]

    X_feat, y = build_dataset(subjects)
    run_training_pipeline(X_feat, y)