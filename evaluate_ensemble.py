import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ১. মডেল ও ডেটা লোড
vectorizer = joblib.load('tfidf_vectorizer_v2.joblib')
df = pd.read_csv('cleaned_spam_emails.csv').sample(200) # পরীক্ষার জন্য ২০০টি স্যাম্পল নিচ্ছি
X_test = vectorizer.transform(df['text'])
y_true = df['label']

# সামঞ্জস্যপূর্ণ মডেলগুলোর লিস্ট
model_names = ['support_vector_machine_svm_best_model_v2.joblib', 'logistic_regression_spam_model.joblib']
models = {name: joblib.load(name) for name in model_names}

# ২. প্রেডিকশন সংগ্রহ
all_preds = {}
for name, model in models.items():
    all_preds[name] = model.predict(X_test)

# ৩. এনসেম্বল (Voting) প্রেডিকশন তৈরি
ensemble_preds = []
for i in range(len(df)):
    votes = [all_preds[name][i] for name in model_names]
    # Majority Voting
    final_vote = 1 if votes.count(1) > votes.count(0) else 0
    ensemble_preds.append(final_vote)

# ৪. রেজাল্ট তুলনা
results = []
def get_metrics(y_true, y_pred, name):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred)
    }

for name in model_names:
    results.append(get_metrics(y_true, all_preds[name], name))
results.append(get_metrics(y_true, ensemble_preds, 'Ensemble (Voting)'))

# ৫. সুন্দর টেবিল আকারে দেখানো
print("\n📊 Model Performance Comparison:")
print(pd.DataFrame(results))