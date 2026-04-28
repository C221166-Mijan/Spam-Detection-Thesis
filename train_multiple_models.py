import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)
from joblib import dump
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Skipping XGBoost model.")

# ============================================================
# 1. Data Loading and Cleaning
# ============================================================
print("=" * 60)
print(" SPAM DETECTION — MULTI-MODEL TRAINING & EVALUATION")
print("=" * 60)

print("\n[1/5] Loading and cleaning data...")

# Try multiple possible dataset paths
for path in ['master_spam_dataset.csv', 'archive/master_spam_dataset.csv']:
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset loaded from: {path}")
        break
    except FileNotFoundError:
        continue
else:
    raise FileNotFoundError("❌ Could not find master_spam_dataset.csv")

# Detect text column
for col in ['message', 'text', 'v2', 'email']:
    if col in df.columns:
        TEXT_COL = col
        break
else:
    TEXT_COL = df.columns[0]
    print(f"⚠️  Using '{TEXT_COL}' as text column.")

# Detect label column
for col in ['label', 'spam', 'class', 'v1']:
    if col in df.columns:
        LABEL_COL = col
        break
else:
    LABEL_COL = df.columns[1]
    print(f"⚠️  Using '{LABEL_COL}' as label column.")

print(f"   Text column  : '{TEXT_COL}'")
print(f"   Label column : '{LABEL_COL}'")
print(f"   Total samples: {len(df)}")

# Ensure label is binary integer (0 = ham, 1 = spam)
if df[LABEL_COL].dtype == object:
    label_map = {}
    unique_labels = df[LABEL_COL].unique()
    for lbl in unique_labels:
        if str(lbl).lower() in ['spam', '1', 'yes']:
            label_map[lbl] = 1
        else:
            label_map[lbl] = 0
    df[LABEL_COL] = df[LABEL_COL].map(label_map)

df = df.dropna(subset=[TEXT_COL, LABEL_COL])
print(f"   Class balance: Ham={( df[LABEL_COL]==0).sum()} | Spam={(df[LABEL_COL]==1).sum()}")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)          # remove emails
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'\d+', '', text)               # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['cleaned'] = df[TEXT_COL].apply(clean_text)

X = df['cleaned']
y = df[LABEL_COL].astype(int)


# ============================================================
# 2. Train / Test Split (80 / 20) — Stratified
# ============================================================
print("\n[2/5] Splitting dataset (80% train / 20% test, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")


# ============================================================
# 3. TF-IDF Vectorization (fit on TRAIN only)
# ============================================================
print("\n[3/5] Vectorizing with TF-IDF N-grams (1,2)...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2
)
X_train_vec = vectorizer.fit_transform(X_train)   # fit only on train
X_test_vec  = vectorizer.transform(X_test)        # transform test separately

dump(vectorizer, 'tfidf_vectorizer_v2.joblib')
print("   ✅ Vectorizer saved: tfidf_vectorizer_v2.joblib")


# ============================================================
# 4. Define Models
# ============================================================
models = {
    'Logistic Regression (LR)':        LogisticRegression(solver='liblinear', C=1.0, random_state=42),
    'Multinomial Naive Bayes (MNB)':   MultinomialNB(alpha=0.1),
    'Support Vector Machine (SVM)':    SVC(kernel='linear', C=1.5, random_state=42),
    'Decision Tree (DT)':              DecisionTreeClassifier(max_depth=20, random_state=42),
    'Random Forest (RF)':              RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
}
if XGBOOST_AVAILABLE:
    models['XGBoost (XGB)'] = XGBClassifier(
        n_estimators=200, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss', random_state=42
    )


# ============================================================
# 5. Train, Evaluate, Compare
# ============================================================
print("\n[4/5] Training and evaluating all models...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_f1   = 0
best_name = None
best_model_obj = None

header = f"{'Model':<35} | {'CV Acc':>7} | {'Test Acc':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}"
print(header)
print("-" * len(header))

for name, clf in models.items():
    # 5-Fold CV on training set
    cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_mean   = cv_scores.mean()

    # Final fit on full training set
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        'Model': name,
        'CV Accuracy': round(cv_mean * 100, 2),
        'Test Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'F1-Score': round(f1 * 100, 2),
    })

    print(f"{name:<35} | {cv_mean*100:>6.2f}% | {acc*100:>7.2f}% | {prec*100:>8.2f}% | {rec*100:>5.2f}% | {f1*100:>5.2f}%")

    # Track best model by F1 on test set
    if f1 > best_f1:
        best_f1        = f1
        best_name      = name
        best_model_obj = clf

print("-" * len(header))
print(f"\n✅ Best Model (by F1-Score): {best_name}  →  F1 = {best_f1*100:.2f}%")


# ============================================================
# 6. Save Best Model + Full Report
# ============================================================
print("\n[5/5] Saving best model and generating report...\n")

safe_name = best_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
dump(best_model_obj, f'{safe_name}_best_model_v2.joblib')
print(f"   ✅ Best model saved: {safe_name}_best_model_v2.joblib")

# Detailed classification report for best model
y_pred_best = best_model_obj.predict(X_test_vec)
print(f"\n{'='*60}")
print(f"  CLASSIFICATION REPORT — {best_name}")
print(f"{'='*60}")
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

# Confusion matrix plot for best model
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title(f'Confusion Matrix: {best_name}\n(Test Set — {len(y_test)} samples)', fontsize=13)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('Actual Label', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=300)
print("   ✅ Confusion matrix saved: confusion_matrix_final.png")

# Model comparison bar chart
results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']
x      = np.arange(len(results_df))
width  = 0.2

fig, ax = plt.subplots(figsize=(13, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results_df[metric], width, label=metric)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Model Comparison — Test Set Performance', fontsize=13)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['Model'], rotation=20, ha='right', fontsize=9)
ax.set_ylim(80, 101)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=300)
print("   ✅ Comparison chart saved: model_comparison_chart.png")

# Save results CSV (thesis table-ready)
results_df.to_csv('model_results.csv', index=False)
print("   ✅ Results table saved: model_results.csv")

print(f"\n{'='*60}")
print(" ALL DONE! Your thesis-ready results are generated.")
print(f"{'='*60}\n")
