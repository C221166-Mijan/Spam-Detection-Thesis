import joblib
import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

print("=" * 65)
print("   ENSEMBLE EVALUATION — All 6 Models + Voting")
print("=" * 65)

# 1. Load data
print("\n[1/4] Loading data...")
df = pd.read_csv('master_spam_dataset.csv')

TEXT_COL  = next((c for c in ['message', 'text', 'v2'] if c in df.columns), df.columns[0])
LABEL_COL = next((c for c in ['label', 'spam', 'v1'] if c in df.columns), df.columns[1])

if df[LABEL_COL].dtype == object:
    lmap = {}
    for lbl in df[LABEL_COL].unique():
        lmap[lbl] = 1 if str(lbl).lower() in ['spam', '1', 'yes'] else 0
    df[LABEL_COL] = df[LABEL_COL].map(lmap)

df = df.dropna(subset=[TEXT_COL, LABEL_COL])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df[TEXT_COL].apply(clean_text)

# 2. Reproduce exact 80/20 test split
print("\n[2/4] Reproducing held-out test set (80/20, seed=42)...")
_, X_test_raw, _, y_test = train_test_split(
    df['cleaned'], df[LABEL_COL].astype(int),
    test_size=0.20, random_state=42, stratify=df[LABEL_COL]
)

vectorizer = joblib.load('tfidf_vectorizer_v2.joblib')
X_test_vec = vectorizer.transform(X_test_raw)
print(f"   Test size: {len(y_test)} | Ham: {(y_test==0).sum()} | Spam: {(y_test==1).sum()}")

# 3. Load all compatible models
print("\n[3/4] Loading all models...")

MODEL_FILES = {
    'Logistic Regression (LR)':      'logistic_regression_lr_best_model_v2.joblib',
    'Multinomial Naive Bayes (MNB)':  'multinomial_naive_bayes_mnb_best_model_v2.joblib',
    'Support Vector Machine (SVM)':   'support_vector_machine_svm_best_model_v2.joblib',
    'Decision Tree (DT)':             'decision_tree_dt_best_model_v2.joblib',
    'Random Forest (RF)':             'random_forest_rf_best_model_v2.joblib',
    'XGBoost (XGB)':                  'xgboost_xgb_best_model_v2.joblib',
}

# Fallback: also try old LR model name
FALLBACKS = {
    'Logistic Regression (LR)': 'logistic_regression_spam_model.joblib',
}

loaded_models = {}
for name, fname in MODEL_FILES.items():
    try:
        loaded_models[name] = joblib.load(fname)
        print(f"   ✅ Loaded: {fname}")
    except FileNotFoundError:
        fallback = FALLBACKS.get(name)
        if fallback:
            try:
                loaded_models[name] = joblib.load(fallback)
                print(f"   ✅ Loaded (fallback): {fallback}")
            except FileNotFoundError:
                print(f"   ⚠️  Skipped (not found): {name}")
        else:
            print(f"   ⚠️  Skipped (not found): {name}")

if not loaded_models:
    print("\n❌ No models found! Run train_multiple_models.py first.")
    exit()

# 4. Evaluate all + ensemble
print("\n[4/4] Evaluating individual models + ensemble...\n")

all_preds = {}
results   = []

for name, clf in loaded_models.items():
    preds = clf.predict(X_test_vec)
    all_preds[name] = preds
    results.append({
        'Model':      name,
        'Accuracy':   round(accuracy_score(y_test, preds) * 100, 2),
        'Precision':  round(precision_score(y_test, preds, zero_division=0) * 100, 2),
        'Recall':     round(recall_score(y_test, preds, zero_division=0) * 100, 2),
        'F1-Score':   round(f1_score(y_test, preds, zero_division=0) * 100, 2),
    })

# Majority Voting Ensemble
preds_matrix = np.array(list(all_preds.values()))   # shape: (n_models, n_samples)
ensemble_preds = (preds_matrix.sum(axis=0) > (len(loaded_models) / 2)).astype(int)

results.append({
    'Model':      f'🗳️ Ensemble Voting ({len(loaded_models)} models)',
    'Accuracy':   round(accuracy_score(y_test, ensemble_preds) * 100, 2),
    'Precision':  round(precision_score(y_test, ensemble_preds, zero_division=0) * 100, 2),
    'Recall':     round(recall_score(y_test, ensemble_preds, zero_division=0) * 100, 2),
    'F1-Score':   round(f1_score(y_test, ensemble_preds, zero_division=0) * 100, 2),
})

# Print table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(results_df))
width  = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for i, m in enumerate(metrics):
    ax.bar(x + i * width, results_df[m], width, label=m)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Ensemble vs Individual Models — Test Set', fontsize=13)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['Model'], rotation=20, ha='right', fontsize=8)
ax.set_ylim(50, 101)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ensemble_comparison_chart.png', dpi=300)
print("\n✅ Chart saved: ensemble_comparison_chart.png")

# Detailed report for ensemble
print(f"\n{'='*65}")
print("   ENSEMBLE DETAILED REPORT")
print(f"{'='*65}")
print(classification_report(y_test, ensemble_preds, target_names=['Ham', 'Spam']))
