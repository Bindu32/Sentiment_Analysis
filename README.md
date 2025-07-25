#  Sentiment Analysis on Social Media Posts

This project analyzes the sentiment of airline-related tweets (positive, neutral, negative) using NLP and machine learning. Built as part of the **TechMaven Project 2**, it demonstrates a full ML pipeline from preprocessing to deployment on Streamlit Cloud.

---

##  Problem Statement

Classify the **sentiment** of social media posts based on their text content.  
Use real-world data to train and deploy a model that predicts whether a tweet is **positive**, **neutral**, or **negative**.

---

##  Live App

👉 [Click here to try the live app](https://sentimentanalysis-aqz86bnujdxpd8ggmvyvmd.streamlit.app/)

---

##  Features

- Real-time text input and sentiment prediction
- Handles noisy tweet data: mentions, URLs, emojis, stopwords
- Clean, minimal Streamlit UI
- Deployed publicly with full ML backend

---

##  Dataset

- **Source**: [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Size: ~14,640 tweets  
- Columns used: `text`, `airline_sentiment`

---

##  Tech Stack

| Component       | Tool                      |
|----------------|---------------------------|
| Language        | Python                    |
| Libraries       | `pandas`, `nltk`, `sklearn` |
| Vectorizer      | TF-IDF                    |
| Classifier      | Logistic Regression       |
| Interface       | Streamlit                 |
| Deployment      | Streamlit Cloud           |
| Hosting Code    | GitHub                    |

---

##  Model Performance

| Metric     | Score   |
|------------|---------|
| Accuracy   | 76.2%   |
| Negative F1 | 0.84   |
| Neutral F1 | 0.58  (improved with class_weight) |
| Positive F1 | 0.71   |

---

##  Workflow Summary

1. Data loading and cleaning
2. Text preprocessing (lemmatization, stopword removal, etc.)
3. TF-IDF vectorization
4. Label encoding
5. Logistic regression training (with class balancing)
6. Model evaluation (Confusion Matrix, F1)
7. Deployment via Streamlit Cloud

---

##  How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/Sentiment-Analysis.git
cd airline-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
```

##  Author
Bindu Sri
- Part of TechMaven Summer Project
- Built with  using open-source tools.

## Contact
- LinkedIn - https://www.linkedin.com/in/bindu-sri-majji-375387258/
- GitHub - https://github.com/Bindu32


