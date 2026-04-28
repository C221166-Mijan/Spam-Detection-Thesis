import joblib
import re
import string

# Load v2 model & vectorizer (same preprocessing as training)
try:
    model      = joblib.load('support_vector_machine_svm_best_model_v2.joblib')
    vectorizer = joblib.load('tfidf_vectorizer_v2.joblib')
    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("   Make sure you have run train_multiple_models.py first.")
    exit()

def clean_text(text):
    """Same cleaning used during training — no NLTK needed."""
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)          # remove emails
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'\d+', '', text)               # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(message: str) -> str:
    cleaned = clean_text(message)
    vector  = vectorizer.transform([cleaned])
    pred    = model.predict(vector)[0]
    return "🚨 SPAM" if pred == 1 else "✅ HAM (Legitimate)"

if __name__ == "__main__":
    print("\n--- Spam Detection System (SVM v2) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter message: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break
        if not user_input:
            continue
        result = predict(user_input)
        print(f"→ {result}\n")
