import streamlit as st
import pickle
import re

# Page config - idi title and icon set chestundi
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🎬")

# 1. Cleaning Function (Idi manam main.py lo vadinde)
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = " ".join(text.split())
    return text

# 2. Loading the Model and Vectorizer
# Ikkada files load avvakapothe error message vastundi
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    st.sidebar.success("✅ AI Brain Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.info("Make sure 'model.pkl' and 'vectorizer.pkl' are in the same folder!")

# 3. UI Design
st.title("🎬 Sentiment Analyzer")
st.write("First AI Project - Trained on 50,000 IMDb Reviews")

user_review = st.text_area("Write your movie review here:", height=150)

if st.button("Analyze Now"):
    if user_review:
        # Preprocess the input
        cleaned = clean_text(user_review)
        vectorized = vectorizer.transform([cleaned])
        
        # Prediction
        prediction = model.predict(vectorized)[0]
        
        # Output with Colors
        if prediction == "positive":
            st.success(f"The AI thinks this is **POSITIVE**! 😊")
            st.balloons() # Chinna celebration!
        else:
            st.error(f"The AI thinks this is **NEGATIVE**! 😡")
    else:
        st.warning("Please type something first, mama!")