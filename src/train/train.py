import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from lightgbm import LGBMClassifier

from src.utils.feature_extraction_duration import get_subj_duration_feature

# ---- build dataset 
X_all = []
y_all = []
for s in range(1, 26):
    subj = f"{s:02d}"
    X_s, y_s = get_subj_duration_feature(subj)
    X_all.append(X_s)
    y_all.append(y_s)

X = pd.concat(X_all, ignore_index=True)
y = np.concatenate(y_all)

# ---- feature matrix ----
x_feat = X[["ied_duration_ms"]]

# ---- k-fold CV ----
metrics = {'accuracy': [], 'precision': [], 'sensitivity': [], 'specificity': [], 'f1': [], 'ROCAUC': [], 'PRAUC': []}
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

for train_index, test_index in kf.split(x_feat, y):
    model = LGBMClassifier()

    x_train_fold, x_test_fold = x_feat.iloc[train_index], x_feat.iloc[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    model.fit(x_train_fold, y_train_fold)

    y_prob = model.predict_proba(x_test_fold)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_test_fold

    metrics['accuracy'].append(accuracy_score(y_true, y_pred))
    metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
    metrics['sensitivity'].append(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    metrics['ROCAUC'].append(roc_auc_score(y_true, y_prob))
    metrics['PRAUC'].append(average_precision_score(y_true, y_prob))

results = pd.DataFrame(metrics)
results.loc['mean'] = results.mean()
print(results)