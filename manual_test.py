import joblib

# ১. তোমার সেরা মডেল এবং ভেক্টরাইজারটি লোড করা হচ্ছে
model = joblib.load('support_vector_machine_svm_best_model_v2.joblib')
vectorizer = joblib.load('tfidf_vectorizer_v2.joblib')

def check_message():
    print("\n--- Spam Detection Testing ---")
    user_input = input("Enter your message to check: ")
    
    # ২. ইনপুট মেসেজটিকে মডেলে দেওয়ার উপযোগী করে তোলা (Transform)
    data = vectorizer.transform([user_input])
    
    # ৩. প্রেডিকশন করা
    prediction = model.predict(data)
    
    # ৪. রেজাল্ট দেখানো
    result = "🚨 SPAM" if prediction[0] == 1 else "✅ HAM (Legitimate)"
    print(f"\nPrediction: {result}\n")

if __name__ == "__main__":
    while True:
        check_message()
        choice = input("Do you want to check another? (y/n): ")
        if choice.lower() != 'y':
            break