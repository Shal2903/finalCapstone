
'''Capstone Project:
This program performs sentiment analysis on customer reviews of customer products.
'''
# Import all the packages including spacy english model and textblob extension. 
import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacytextblob')

# Load amazon data, remove nan values from reviews. 
amazon_data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv', sep =',', low_memory=False)
amazon_data = amazon_data.dropna(subset=['reviews.text'])

# Selected specific review from the amazon data using Indexing.
reviews_data = amazon_data['reviews.text'].iloc[[15, 23, 55, 340, 450, 1500, 1789, 2200, 2500]]


# Funtion to preprocess text data by converting to lower case, removing stopwords and leading / trailing white spaces.
def preprocess_text(reviews):
    doc = nlp(reviews)
    cleaned_text = [token.text.lower() for token in doc if not token.is_stop and token.is_punct]
    return " ".join(cleaned_text)

# Function to perform senitment analysis on cleaned data using polarity attribute. 
def sentiment_analysis(cleaned_data):
    for key, value in cleaned_data.items():
        doc = nlp(value)
        print(f"Review: {key, value}")
        print(f"Sentiment: {doc._.blob.polarity}\n")


# Perform sentiment analysis on the data that is cleaned and reviewed. 
print("*******Sentiment Analyis on Consumer Review of Amazon Products!*******\n")
sentiment_analysis(reviews_data)

# Select two sentences from the data set using Indexing.
first_review = amazon_data['reviews.text'][1500]
second_review = amazon_data['reviews.text'][2200]

# Funtion to test similarity betweeen the two selected sentences using similarity function. 
def similarity(first, second):
    similarity_result = nlp(first).similarity(nlp(second))
    return(similarity_result)

# Output the results of sentiment analysis

print(f'\nFirst Review : {first_review}')
print(f'\nSecond Review : {second_review}')
print (f'\nSimilarity between the first and second review: {similarity(first_review, second_review)}')