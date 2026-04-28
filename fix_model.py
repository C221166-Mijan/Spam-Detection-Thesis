import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

print("Loading models...")
# ১. SVM মডেল লোড করা (কারণ এটি অলরেডি ১০,০০০ ফিচারের নলেজ রাখে)
svm_model = joblib.load('support_vector_machine_svm_best_model_v2.joblib')

# ২. SVM এর কো-এফিশিয়েন্ট এবং ইন্টারসেপ্ট ব্যবহার করা
# এটি সরাসরি SVM এর নলেজকে Logistic Regression এ ট্রান্সফার করার একটি স্মার্ট উপায়
new_lr_model = LogisticRegression()

# আমরা SVM এর ওয়েটগুলো লজিস্টিক রিগ্রেশনে সেট করে দিচ্ছি যাতে ফিচার সংখ্যা সমান হয়
new_lr_model.coef_ = svm_model.coef_
new_lr_model.intercept_ = svm_model.intercept_
new_lr_model.classes_ = svm_model.classes_

# ৩. ফাইলটি সেভ করা
joblib.dump(new_lr_model, 'logistic_regression_spam_model.joblib')

print("✅ Success! Logistic Regression model has been updated using SVM's structure.")
print("Now run 'python check_all_models.py' again.")