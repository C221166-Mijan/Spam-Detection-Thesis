import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
from joblib import load

print("=" * 60)
print("   FINAL EVALUATION — SVM (Best Model)")
print("=" * 60)

# 1. Load data & models
print("\n[1/3] Loading data and best model...")
df = pd.read_csv('master_spam_dataset.csv')
model      = load('support_vector_machine_svm_best_model_v2.joblib')
vectorizer = load('tfidf_vectorizer_v2.joblib')

# Detect columns
TEXT_COL  = next((c for c in ['message', 'text', 'v2'] if c in df.columns), df.columns[0])
LABEL_COL = next((c for c in ['label', 'spam', 'v1'] if c in df.columns), df.columns[1])
print(f"   Text: '{TEXT_COL}' | Label: '{LABEL_COL}' | Total: {len(df)}")

# Ensure binary labels
if df[LABEL_COL].dtype == object:
    lmap = {}
    for lbl in df[LABEL_COL].unique():
        lmap[lbl] = 1 if str(lbl).lower() in ['spam', '1', 'yes'] else 0
    df[LABEL_COL] = df[LABEL_COL].map(lmap)

df = df.dropna(subset=[TEXT_COL, LABEL_COL])

# Text cleaning (same as train_multiple_models.py)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df[TEXT_COL].apply(clean_text)

# Reproduce EXACT same 80/20 split (random_state=42, stratified)
print("\n[2/3] Reproducing 80/20 test split (stratified, seed=42)...")
_, X_test_raw, _, y_test = train_test_split(
    df['cleaned'], df[LABEL_COL].astype(int),
    test_size=0.20, random_state=42, stratify=df[LABEL_COL]
)
X_test_vec = vectorizer.transform(X_test_raw)
print(f"   Test samples: {len(y_test)} (Ham: {(y_test==0).sum()} | Spam: {(y_test==1).sum()})")

# 3. Evaluate
print("\n[3/3] Evaluating SVM on held-out test set...")
y_pred = model.predict(X_test_vec)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"   FINAL CLASSIFICATION REPORT — SVM")
print(f"{'='*60}")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print(f"   Accuracy  : {acc*100:.2f}%")
print(f"   Precision : {prec*100:.2f}%")
print(f"   Recall    : {rec*100:.2f}%")
print(f"   F1-Score  : {f1*100:.2f}%")
print(f"{'='*60}")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Ham (Real)', 'Spam'],
            yticklabels=['Ham (Real)', 'Spam'])
plt.title(f'Confusion Matrix: SVM on Held-Out Test Set ({len(y_test)} samples)', fontsize=13)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=300)
print("\n✅ Confusion matrix saved: confusion_matrix_final.png")
