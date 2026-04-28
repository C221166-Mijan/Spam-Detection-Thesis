import joblib
import os
import warnings

# অপ্রয়োজনীয় ওয়ার্নিং (Version mismatch) হাইড করার জন্য
warnings.filterwarnings("ignore")

# ১. ভেক্টরাইজার লোড করা
try:
    vectorizer = joblib.load('tfidf_vectorizer_v2.joblib')
    EXPECTED_FEATURES = 10000 
except:
    print("❌ Vectorizer not found!")
    exit()

# ২. মডেল ফাইলগুলো খুঁজে বের করা
all_files = os.listdir()
model_files = [f for f in all_files if f.endswith('.joblib') and 'vectorizer' not in f]

print(f"🔍 Found {len(model_files)} model files. Analyzing...")

def check_with_all_models(text):
    text_vector = vectorizer.transform([text])
    votes = []
    
    print(f"\nMessage: {text}")
    print(f"{'Model Name':<45} | {'Status/Prediction'}")
    print("-" * 75)
    
    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            
            # মডেলের ফিচার সংখ্যা চেক করা (এটিই তোমার এরর ফিক্স করবে)
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            elif hasattr(model, 'coef_'):
                n_features = model.coef_.shape[1]
            else:
                n_features = EXPECTED_FEATURES 

            # যদি ফিচার ম্যাচ না করে তবে স্কিপ করবে
            if n_features != EXPECTED_FEATURES:
                print(f"{model_file:<45} | ⚠️ Skipped (Feature Mismatch: {n_features})")
                continue

            # প্রেডিকশন
            prediction = model.predict(text_vector)[0]
            label = "🚨 SPAM" if prediction == 1 else "✅ HAM"
            votes.append(prediction)
            print(f"{model_file:<45} | {label}")

        except Exception as e:
            print(f"{model_file:<45} | ❌ Error Loading")

    # ফাইনাল সিদ্ধান্ত
    if votes:
        final_result = "🚨 SPAM" if votes.count(1) > votes.count(0) else "✅ HAM"
        print("-" * 75)
        print(f"📊 Final Consensus: {final_result} (Spam: {votes.count(1)}, Ham: {votes.count(0)})")
    else:
        print("\n❌ No compatible models available for this message.")

if __name__ == "__main__":
    while True:
        user_msg = input("\nEnter message (or 'exit'): ")
        if user_msg.lower() == 'exit': break
        check_with_all_models(user_msg)