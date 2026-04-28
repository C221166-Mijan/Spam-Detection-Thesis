import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from joblib import load

# ১. ডাটা এবং মডেল লোড করা
print("Loading data and models...")
df = pd.read_csv('master_spam_dataset.csv')
model = load('support_vector_machine_svm_best_model_v2.joblib')
vectorizer = load('tfidf_vectorizer_v2.joblib')

# ২. কলামের সঠিক নাম খুঁজে বের করা
# আমরা চেক করছি 'text' বা 'message' এর মধ্যে কোনটা তোমার ফাইলে আছে
if 'text' in df.columns:
    column_name = 'text'
elif 'message' in df.columns:
    column_name = 'message'
elif 'v2' in df.columns:
    column_name = 'v2'
else:
    # যদি কোনোটিই না মেলে, তবে প্রথম কলামটিকেই টেক্সট ধরে নেবে (সাধারণত তাই হয়)
    print(f"⚠️ Warning: Common column names not found. Using '{df.columns[0]}' as text column.")
    column_name = df.columns[0]

print(f"✅ Using '{column_name}' as the message column.")

# ৩. প্রেডিকশন তৈরি
X_vectors = vectorizer.transform(df[column_name].astype(str))
y_true = df['label']
y_pred = model.predict(X_vectors)

# ৪. Confusion Matrix তৈরি
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Ham (Real)', 'Spam'], 
            yticklabels=['Ham (Real)', 'Spam'])

plt.title(f'Confusion Matrix: SVM Performance on {len(df)} Messages', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)

# গ্রাফ সেভ করা
plt.savefig('confusion_matrix_final.png', dpi=300)
print("✅ Confusion Matrix saved as 'confusion_matrix_final.png'")

# ৫. ফাইনাল রিপোর্ট
print("\n" + "="*50)
print("       FINAL CLASSIFICATION REPORT (SVM)")
print("="*50)
print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))
print(f"Final Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
print("="*50)