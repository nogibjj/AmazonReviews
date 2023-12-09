#read the token_count file and generate the tokens

import pandas as pd
import numpy as np

#read the token_count file and store it in a dataframe

probs = pd.read_csv('token_count_model_probabilities.csv', sep=',')

print (probs)


#save the reviews_sample_stratified file in a dataframe

reviews = pd.read_csv('reviews_sample_stratified.csv', sep=',')

print (reviews)

#change sentiment column positive to Positive and negative to Negative
reviews['sentiment'] = reviews['sentiment'].replace({'positive': 'Positive', 'negative': 'Negative'})

print (reviews)

#turn everything in reviewText to str
reviews['reviewText'] = reviews['reviewText'].astype(str)

#turn everything in probs to str
probs['Word'] = probs['Word'].astype(str)


######### Generate Synthetic Reviews from token count model #########
for index, row in reviews.iterrows():
    sentiment = row['sentiment']
    review_length = len(row['reviewText'].split())
    
    # Generate a synthetic review by randomly selecting words from the probs dataframe
    # The selection is weighted based on the probabilities in the 'Positive' or 'Negative' column
    synthetic_review = " ".join(np.random.choice(probs['Word'], size=review_length, p=probs[sentiment]))
    
    #add the synthetic review to the dataframe
    reviews.loc[index, 'synthetic_review'] = synthetic_review


#just keep the columns we need (reviewText, sentiment, synthetic_review)

reviews = reviews[['sentiment', 'synthetic_review']]

#change the name of the column synthetic_review to reviewText

reviews = reviews.rename(columns={'synthetic_review': 'reviewText'})

#save the reviews dataframe to a csv file

reviews.to_csv('synthetic_prob_count.csv')

######### Generate Synthetic Reviews from tfidf model #########
probs_tfdf = pd.read_csv('tfidf_model_probabilities.csv', sep=',')



#turn everything in probs_tfdf to str
probs_tfdf['Word'] = probs_tfdf['Word'].astype(str)

for index, row in reviews.iterrows():
    sentiment = row['sentiment']
    review_length = len(row['reviewText'].split())
    
    # Generate a synthetic review by randomly selecting words from the probs dataframe
    # The selection is weighted based on the probabilities in the 'Positive' or 'Negative' column
    synthetic_review = " ".join(np.random.choice(probs_tfdf['Word'], size=review_length, p=probs_tfdf[sentiment]))
    
    #add the synthetic review to the dataframe
    reviews.loc[index, 'synthetic_review_tfidf'] = synthetic_review


#just keep the columns we need (reviewText, sentiment, synthetic_review_tfidf)

reviews = reviews[['sentiment', 'synthetic_review_tfidf']]

#change the name of the column synthetic_review_tfidf to reviewText

reviews = reviews.rename(columns={'synthetic_review_tfidf': 'reviewText'})

reviews.to_csv('synthetic_tfidf.csv')