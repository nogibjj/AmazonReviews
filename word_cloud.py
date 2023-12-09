from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    print(21)
    df_probabilities = pd.read_csv('tfidf_model_probabilities.csv')

    # Create a dictionary of words and their probabilities
    word_probs = {word: prob for word, prob in zip(df_probabilities['Word'], 
                                                df_probabilities['Positive'])}

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white')

    # Generate a word cloud
    wordcloud.generate_from_frequencies(word_probs)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    # save the image
    plt.savefig('real_data_word_cloud_positive.png')
