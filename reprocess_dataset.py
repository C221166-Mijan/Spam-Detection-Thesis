import pandas as pd

# File load
df = pd.read_csv("C:\\SPAM DETECTION\\dataset\\spam.txt", encoding='latin-1')

# Check first few rows
print(df.head())
