#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import requests
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import scipy.linalg as linalg

# Define the web scraping function
def scrape_reviews():
    wm_title = []
    wm_date = []
    wm_content = []
    wm_rating = []

    for i in range(1, 150):
        link = f"https://www.amazon.in/Apple-MacBook-Air-13-3-inch-MQD32HN/product-reviews/B073Q5R6VR/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={i}"
        response = requests.get(link)
        soup = bs(response.content, "html.parser")

        # Extracting Review Title
        title = soup.find_all('a', class_='review-title-content')
        review_title = [title[i].get_text().strip() for i in range(len(title))]
        wm_title.extend(review_title)

        # Extracting Ratings
        rating = soup.find_all('i', class_='review-rating')
        review_rating = [rating[i].get_text().rstrip(' out of 5 stars') for i in range(2, len(rating))]
        wm_rating.extend(review_rating)

        # Extracting Content of review
        review = soup.find_all("span", {"data-hook": "review-body"})
        review_content = [review[i].get_text().strip() for i in range(len(review))]
        wm_content.extend(review_content)

        # Extracting dates of reviews
        dates = soup.find_all('span', class_='review-date')
        review_dates = [dates[i].get_text().lstrip('Reviewed in India on') for i in range(2, len(rating))]
        wm_date.extend(review_dates)

    df = pd.DataFrame({'Title': wm_title, 'Ratings': wm_rating, 'Comments': wm_content, 'Date': wm_date})
    df['Date'] = pd.to_datetime(df['Date'])
    df['Ratings'] = df['Ratings'].astype(float)
    return df

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    return " ".join(text)

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function to show wordcloud
def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(str(data))

    fig = plt.figure(1, figsize=(20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Amazon Product Reviews Analysis")

    st.write("Scraping and Analyzing Amazon Product Reviews")

    if st.button("Scrape Reviews"):
        df = scrape_reviews()
        st.write("Scraped Reviews")
        st.write(df.head())

        # Clean text data
        df["Comments"] = df["Comments"].apply(lambda x: clean_text(x))
        df["Title"] = df["Title"].astype(str)
        df["Title"] = df["Title"].apply(lambda x: clean_text(x))

        # Sentiment Analysis
        sid = SentimentIntensityAnalyzer()
        df["sentiments"] = df["Comments"].apply(lambda x: sid.polarity_scores(x))
        df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

        # Additional features
        df["nb_chars"] = df["Comments"].apply(lambda x: len(x))
        df["nb_words"] = df["Comments"].apply(lambda x: len(x.split(" ")))

        # Doc2Vec
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["Comments"].apply(lambda x: x.split(" ")))]
        model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
        doc2vec_df = df["Comments"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
        doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
        df = pd.concat([df, doc2vec_df], axis=1)

        # TF-IDF
        tfidf = TfidfVectorizer(min_df=10)
        tfidf_result = tfidf.fit_transform(df["Comments"]).toarray()
        tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names_out())
        tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
        tfidf_df.index = df.index
        df = pd.concat([df, tfidf_df], axis=1)

        st.write("Data after feature engineering")
        st.write(df.head())

        st.write("Wordcloud of Comments")
        show_wordcloud(df["Comments"])

        st.write("Wordcloud of Titles")
        show_wordcloud(df["Title"])

        st.write("Highest positive sentiment reviews (with more than 5 words)")
        st.write(df[df["nb_words"] >= 5].sort_values("pos", ascending=False)[["Comments", "pos"]].head(10))

        st.write("Lowest negative sentiment reviews (with more than 5 words)")
        st.write(df[df["nb_words"] >= 5].sort_values("neg", ascending=False)[["Comments", "neg"]].head(10))

if __name__ == "__main__":
    main()

