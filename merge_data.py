import pandas as pd

# ১. আগের SMS ডেটাসেট লোড ও ক্লিন করা
df1 = pd.read_csv('archive/spam.csv', encoding='latin-1')
df1 = df1[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df1['label'] = df1['label'].map({'ham': 0, 'spam': 1})

# ২. নতুন Email ডেটাসেট লোড ও ক্লিন করা (তোমার আপলোড করা ফাইল)
df2 = pd.read_csv('archive/spam_ham_dataset.csv')
# এই ফাইলে কলাম নাম থাকে 'label_num' এবং 'text'
df2 = df2[['label_num', 'text']].rename(columns={'label_num': 'label', 'text': 'message'})

# ৩. দুইটা ডাটাফ্রেমকে একত্রে করা
combined_df = pd.concat([df1, df2], ignore_index=True)

# ৪. ডুপ্লিকেট মেসেজ থাকলে তা মুছে ফেলা
combined_df = combined_df.drop_duplicates(subset=['message'])

# ৫. ফাইনাল ফাইলটি সেভ করা
combined_df.to_csv('archive/master_spam_dataset.csv', index=False)

print(f"Dataset Merged Successfully!")
print(f"Total samples in Master Dataset: {len(combined_df)}")