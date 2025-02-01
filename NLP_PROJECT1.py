#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")
from transformers import pipeline

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result["label"], result["score"]

if __name__ == "__main__":
    text = input("Enter text: ")
    label, score = analyze_sentiment(text)
    print(f"Sentiment: {label} (Confidence: {score:.2f})")

