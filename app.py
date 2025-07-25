import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load trained model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Streamlit UI
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.write("Enter a tweet and find out if it's positive, neutral, or negative!")

user_input = st.text_area("Tweet Text")

if st.button("Analyze"):
    cleaned = preprocess_text(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    sentiment = le.inverse_transform(prediction)[0]
    st.subheader(f"Predicted Sentiment: **{sentiment.upper()}**")
