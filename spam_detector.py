import pandas as pd
import re
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. মডেল এবং ভেক্টরাইজার লোডিং ---

# The best model (SVM) found from cross-validation is loaded.
try:
    model = load('support_vector_machine_svm_best_model.joblib')
    print("Loaded the best model: Support Vector Machine (SVM)")
except FileNotFoundError:
    print("Error: SVM model file not found. Ensure 'support_vector_machine_svm_best_model.joblib' is in the folder.")
    exit()

# Load the fitted TF-IDF Vectorizer
try:
    vectorizer = load('tfidf_vectorizer_final.joblib')
    print("Loaded TF-IDF Vectorizer.")
except FileNotFoundError:
    print("Error: TF-IDF Vectorizer file not found.")
    exit()

# --- 2. টেক্সট প্রসেসিং ফাংশন ---

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and processes the input text for prediction."""
    # Remove special characters and single characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    
    # Substitute multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    # Convert to lower case and remove stop words
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    
    return text

# --- 3. প্রেডিকশন ফাংশন ---

def predict_spam(message):
    """Processes message and makes prediction using the loaded SVM model."""
    processed_message = preprocess_text(message)
    
    # Transform the single message using the loaded vectorizer
    message_vector = vectorizer.transform([processed_message])
    
    # Make prediction
    prediction = model.predict(message_vector)
    
    if prediction[0] == 1:
        return "SPAM"
    else:
        return "HAM (Not Spam)"

# --- 4. মেইন লুপ (ইউজার ইন্টারফেস) ---

if __name__ == "__main__":
    print("\n--- SPAM DETECTION SYSTEM (Using SVM Model) ---")
    print("Enter 'exit' or 'quit' to stop the system.")
    
    while True:
        user_input = input("\nEnter a message to check: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("System stopped. Thank you!")
            break
            
        if user_input.strip() == "":
            continue

        result = predict_spam(user_input)
        
        print(f"\n--- Prediction Result ---")
        print(f"Message: {user_input}")
        print(f"Result: {result}")
        print("-------------------------")