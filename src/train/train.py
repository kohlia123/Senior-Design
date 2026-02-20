import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# Internal imports
from src.utils.preprocessing import get_subj_data
from src.config import N_SUB

def build_dataset(subjects):
    # This runs get_subj_data for multiple subjects in parallel
    results = Parallel(n_jobs=-1)(delayed(get_subj_data)(s) for s in subjects)
    
    X_all = [res[0] for res in results]
    y_all = [y for res in results for y in res[1]]
    
    X = pd.concat(X_all, axis=0, ignore_index=True)
    y = np.asarray(y_all, dtype=int)
    return X, y

def run_training_pipeline(X, y, n_splits=5, random_state=15):
    metrics = {
        "accuracy": [], "precision": [], "sensitivity": [], "specificity": [],
        "f1": [], "ROCAUC": [], "PRAUC": [], "train_time_s": []
    }

    groups = X['subj']
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rus = RandomUnderSampler(random_state=random_state)

    all_shap_values = []
    all_test_features = []
    
    feature_cols = [c for c in X.columns if c not in ['subj', 'chan_name', 'epoch', 'epoch_id']]

    for fold, (train_index, test_index) in enumerate(kf.split(X, y, groups=groups)):
        print(f"--- Processing Fold {fold + 1} ---")
        try:
            X_train_raw = X.iloc[train_index][feature_cols]
            X_test_raw = X.iloc[test_index][feature_cols]
            y_train_raw, y_test = y[train_index], y[test_index]

            X_train, y_train = rus.fit_resample(X_train_raw, y_train_raw)

            # add class_weight='balanced'
            model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                n_jobs=-1,
                random_state=random_state
            )
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start
            print(f"  Fold {fold + 1} training time: {train_time:.2f}s")

            y_score = model.predict_proba(X_test_raw)[:, 1]

            # find best threshold using precision-recall curve
            from sklearn.metrics import precision_recall_curve
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_train_raw, 
                                                        model.predict_proba(X_train_raw)[:, 1])
            f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-9)
            best_thresh = thresholds[np.argmax(f1_scores[:-1])]  # thresholds is 1 shorter than p/r
            print(f"  Best threshold for fold {fold + 1}: {best_thresh:.3f}")

            # use best threshold instead of default 0.5
            y_pred = (y_score >= best_thresh).astype(int)

            # Metrics 
            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            metrics["sensitivity"].append(recall_score(y_test, y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            metrics["train_time_s"].append(train_time)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            metrics["specificity"].append(tn / (tn + fp) if (tn + fp) else 0.0)
            
            if len(np.unique(y_test)) == 2:
                metrics["ROCAUC"].append(roc_auc_score(y_test, y_score))
                metrics["PRAUC"].append(average_precision_score(y_test, y_score))

            # SHAP 
            explainer = shap.TreeExplainer(model)
            X_test_sample = X_test_raw.sample(min(len(X_test_raw), 100), random_state=random_state)
            shap_v = explainer.shap_values(X_test_sample)
            print(f"  SHAP type: {type(shap_v)}, shape: {np.array(shap_v).shape}")
            if isinstance(shap_v, list):
                all_shap_values.append(shap_v[1])
            else:
                all_shap_values.append(shap_v[:, :, 1])
            all_test_features.append(X_test_sample)
            print(f"  Fold {fold + 1} complete. SHAP samples collected: {len(X_test_sample)}")

        except Exception as e:
            print(f"  Fold {fold + 1} FAILED with error: {e}")
            raise

    # Summary (unchanged)
    results = pd.DataFrame(metrics)
    results.loc["mean"] = results.mean(numeric_only=True)
    print("\nFinal Results Table:")
    print(results)

    print("\nGenerating SHAP Feature Importance Plot...")
    print(f"Folds completed: {len(all_test_features)}")

    if len(all_test_features) == 0:
        print("ERROR: No SHAP values were collected. Check fold errors above.")
        return results

    final_shap = np.vstack(all_shap_values)
    final_features = pd.concat(all_test_features, ignore_index=True)
    final_features = final_features[feature_cols]

    print(f"  final_shap shape: {final_shap.shape}")
    print(f"  final_features shape: {final_features.shape}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(final_shap, final_features, plot_type='bar', color='#b0acf7')
    plt.show()

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/trained_model.joblib")
    print("Model saved to models/trained_model.joblib")

    return results

if __name__ == "__main__":
    # Load list of subjects (e.g., ['01', '02', ...])
    subjects = [f"{i:02d}" for i in range(1, N_SUB + 1)]
    
    # Build dataset
    X_feat, y = build_dataset(subjects)

    # Run pipeline
    run_training_pipeline(X_feat, y)