import joblib

# ১. তোমার সেরা মডেল এবং ভেক্টরাইজারটি লোড করা হচ্ছে
model = joblib.load('support_vector_machine_svm_best_model_v2.joblib')
vectorizer = joblib.load('tfidf_vectorizer_v2.joblib')

def check_message():
    print("\n--- Spam Detection Testing ---")
    
    # Message খালি না হওয়া পর্যন্ত জিজ্ঞাসা করা
    user_input = input("Enter your message to check: ").strip()
    
    # যদি খালি হয় তাহলে exit করা
    if not user_input:
        return False
    
    # २. ইনপুট মেসেজটিকে মডেলে দেওয়ার উপযোগী করে তোলা (Transform)
    data = vectorizer.transform([user_input])
    
    # ३. প্রেডিকশন করা
    prediction = model.predict(data)
    
    # ४. রেজাল্ট দেখানো
    result = "🚨 SPAM" if prediction[0] == 1 else "✅ HAM (Legitimate)"
    print(f"\nPrediction: {result}")
    
    return True  # চলতে থাকা

if __name__ == "__main__":
    while True:
        if not check_message():
            print("\nThank you! Exiting...\n")
            break