import pandas as pd

df = pd.read_csv("IMDB Dataset.csv")

print("----First 5 rows----")
print(df.head())

print("\n----Sentiment Counts----")
print(df["sentiment"].value_counts())

import re

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove extra spaces
    text = " ".join(text.split())
    return text

print("\nCleaning data... please wait (it might take a minute)")
df["Cleaned_Review"] = df["review"].apply(clean_text)

print("\n--- Cleaned vs Original ---")
print("Original Review:", df['review'][1][:100]) # First 100 Characters
print("Cleaned Review:", df['Cleaned_Review'][1][:100]) # First 100 Characters

from sklearn.model_selection import train_test_split

# 1. Inputs (X) and Target (y)
X = df['Cleaned_Review'] # Clean reviews
y = df['sentiment']      # Labels (positive/negative)

# 2. Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Vectorization (Words -> Numbers)
print("\nConverting text to numbers... please wait.")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 2. Model Training (The AI Brain)
print("Training the AI model (Logistic Regression)...")
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 3. Quick Evaluation (Testing the brain)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Training Complete!")
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred))

import pickle

# 1. Save the Model (The logic)
pickle.dump(model, open("model.pkl", "wb"))

# 2. Save the Vectorizer (The vocabulary)
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n💾 Model and Vectorizer saved successfully as .pkl files!")

