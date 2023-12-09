"""This script performs sentiment analysis using the Naive Bayes algorithm."""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess the text by removing punctuation, numbers, stopwords, and
    lemmatizing the words."""
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a single string
    text = ' '.join(words)
    return text


def naive_bayes_sentiment_analysis(df, test_size=0.2, random_state=22, tfidf=False):
    """Perform sentiment analysis using the Naive Bayes algorithm."""
    # Ensure the necessary columns are present
    str_ = "DataFrame must contain 'reviewText' and 'sentiment' columns"
    assert 'reviewText' in df.columns and 'sentiment' in df.columns, str_

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], 
                                                        df['sentiment'], 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    """The TfidfVectorizer and CountVectorizer are classes provided by the 
    sklearn.feature_extraction.text module in the scikit-learn library. 
    
    TfidfVectorizer: This class is used to convert a collection of raw documents 
    to a matrix of TF-IDF features. TF-IDF stands for Term Frequency-Inverse Document 
    Frequency, a numerical statistic that reflects how important a word is to a 
    document in a collection or corpus.

    CountVectorizer: This class is used to convert a collection of text documents to 
    a matrix of token counts. It implements both tokenization and occurrence counting 
    in a single class.

    Both of these classes are used for feature extraction in text data preprocessing. 
    They convert text data into a form that can be used by machine learning 
    algorithms."""

    if tfidf:
        # Convert the reviews into a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        extractor = "TF-IDF"
    else:
        # Convert the reviews into a matrix of token counts
        vectorizer = CountVectorizer()
        extractor = "Token Counts"
    
    X_train_counts = vectorizer.fit_transform(X_train)

    # Train the Naive Bayes model
    clf = MultinomialNB().fit(X_train_counts, y_train)

    # Get the log probabilities of each word
    log_probabilities = clf.feature_log_prob_
    probabilities = np.exp(log_probabilities)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame
    df_probabilities = pd.DataFrame(probabilities.T, 
                                    columns=['Negative', 'Positive'])
    df_probabilities['Word'] = feature_names

    df_log_probs = pd.DataFrame(log_probabilities.T, 
                                columns=['Negative', 'Positive'])
    df_log_probs['Word'] = feature_names

    # Transform the test data into a matrix of token counts
    X_test_counts = vectorizer.transform(X_test)

    # Make predictions on the test data
    y_pred = clf.predict(X_test_counts)

    # Print the accuracy of the model
    print(f"\n{'-'*40}")
    print(f"Feature Extraction Method: {extractor}")
    print(f"{'-'*40}")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {metrics.precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Recall: {metrics.recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {metrics.f1_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"{'-'*40}\n")

    return df_probabilities, df_log_probs


if __name__  == '__main__':
    print("\nReal Data - Raw")
    df = pd.read_csv('reviews_sample_stratified.csv')
    df = df[['reviewText', 'sentiment']]
    df = df.dropna()
    # Preprocess the text
    # df['reviewText'] = df['reviewText'].apply(preprocess_text)
    prob_1, p1_log = naive_bayes_sentiment_analysis(df)
    prob_2, p2_log = naive_bayes_sentiment_analysis(df, tfidf=True)

    print("\nReal Data - Preprocessed")
    df = pd.read_csv('reviews_sample_stratified_preprocessed.csv')
    df = df[['reviewText', 'sentiment']]
    df = df.dropna()
    # Preprocess the text
    # df['reviewText'] = df['reviewText'].apply(preprocess_text)
    prob_1, p1_log = naive_bayes_sentiment_analysis(df)
    prob_2, p2_log = naive_bayes_sentiment_analysis(df, tfidf=True)

    print("\nGenerated data - token counts")
    df = pd.read_csv('synthetic_prob_count.csv')
    df = df[['reviewText', 'sentiment']]
    df = df.dropna()
    # Preprocess the text
    # df['reviewText'] = df['reviewText'].apply(preprocess_text)
    prob_1, p1_log = naive_bayes_sentiment_analysis(df)
    prob_2, p2_log = naive_bayes_sentiment_analysis(df, tfidf=True)

    print("\ngenerated data - tfidf")
    df = pd.read_csv('synthetic_tfidf.csv')
    df = df[['reviewText', 'sentiment']]
    df = df.dropna()
    # Preprocess the text
    # df['reviewText'] = df['reviewText'].apply(preprocess_text)
    prob_1, p1_log = naive_bayes_sentiment_analysis(df)
    prob_2, p2_log = naive_bayes_sentiment_analysis(df, tfidf=True)

    print(21)
