# Sentiment Analysis on Amazon Electronics Reviews

## Overview

This repository contains code and documentation for a sentiment analysis project on Amazon electronics reviews. The project focuses on classifying reviews as positive or negative using both generative (Naive Bayes) and discriminative (BERT) approaches. The dataset consists of 50,000 reviews evenly split between positive and negative sentiments.

## Project Structure

- **Data:** The dataset used in this project is obtained from the Amazon product review dataset curated by J. McAuley, Jianmo Ni, and Jiacheng Li at UCSD. It is available for public research use. The dataset is specifically from the electronics category, with preprocessing details outlined in the report.

- **Models:**
  - **Generative Model (Naive Bayes):** The Naive Bayes classifier is implemented using the scikit-learn library. The model is trained on real data and used to generate synthetic data for comparison.
  - **Discriminative Model (BERT):** The BERT model, based on Bidirectional Encoder Representations from Transformers, is implemented using the Hugging Face's transformers library. It is fine-tuned for sentiment analysis on the Amazon electronics reviews.

## Results and Discussion

The sentiment analysis conducted on Amazon electronics reviews using the Naive Bayes and BERT models yielded insightful results, particularly when comparing their performance on real and synthetic data.

The Naive Bayes model achieved an accuracy of 83.82% on real data, while its performance on synthetic data was notably higher at 87.85%. On the other hand, the BERT model demonstrated superior accuracy on real data with 92.40%, but a slightly reduced performance of 89.28% on synthetic data.

The Naive Bayes model's performed better on synthetic data compared to real data. This can be attributed to the nature of the synthetic data. It was generated based on the log probabilities obtained from the Naive Bayes model trained on the real dataset. These probabilities inherently delineate clear boundaries between words that belong to different sentiment classes. As a result, the synthetic data aligns more closely with the assumptions and statistical model of the Naive Bayes classifier. This alignment makes the synthetic data inherently easier for the Naive Bayes algorithm to classify accurately, as the data is almost tailor-made for the model's probabilistic approach.

The BERT model performed better on real data compared to synthetic data. This difference in performance stems from the inherent ability of discriminative models in capturing the intricate patterns and subtle nuances present in authentic customer reviews, allowing them to navigate the complexities in genuine sentiments. In contrast, synthetic data lacks the richness and variability found in real-world scenarios, making it challenging to accurately replicate and interpret the intricate features of actual customer sentiments. The bidirectional attention mechanism, unique to the BERT model, enables it to analyze the entire context of a word by considering its relationship with all other words in the sequence. In this way, it is able to capture complex dependencies and contextual relationships within a sentence or document to extract meaningful insights from the intricacies of real customer feedback, thus contributing to its effectiveness in this context.

**Model** | **Real Data (Accuracy)** | **Synthetic Data (Accuracy)**
--- | --- | ---
Generative Model (Naive Bayes) | 83.82% | 87.85%
Discriminative Model (BERT) | 92.40% | 89.28%

### Future Work

One limitation of our analysis and the presented analysis is the generation of our synthetic data, which makes use of our preprocessed data and the token probabilities obtained from the TF-IDF vectorizer, undermining the impact of stop words while generating data. This makes our generated data slightly different from conventional English sentences, making our analysis a bit unrealistic on synthetic data. A combination of TF-IDF and n-grams would have been a better approach, which would have retained the order of a few words together, making the vocabulary more English-like. We can also do the same analysis with a more powerful GPU and more memory size and thus can use the full dataset (more than 6 million reviews), which can help us better train the model and improve our analysis and better distinguish between our discriminative and generative approach. This can allow us to also explore reviews for other categories on Amazon, to see if there are any changes when working with data from other categories.

Another recommendation for improving this project could be to make this problem multinomial, and not only predict “positive” vs “negative” but rather predict the rating itself, from 1-5, making this a multi-class classification problem. We can also explore the dataset with the original proportions of positive and negative reviews, i.e., analyze the imbalanced dataset to see what results we get while training the model on an imbalanced dataset.

### Conclusion

This report provides a comprehensive view of the application and efficacy of natural language processing (NLP) models in sentiment analysis, with a specific focus on analyzing 50,000 customer reviews (50% positive, 50% negative) in the electronics sector on Amazon while comparing two approaches: generative and discriminative.

In our study, both approaches were applied to real customer reviews as well as synthetic data generated from a Naive Bayes Model. This synthetic data was designed to replicate the distribution properties observed in the actual reviews.

The discriminative method achieved an accuracy of 92.4% on the real reviews while the generative method achieved an accuracy of 87.5% on the synthetic reviews. This comparison showcases the strengths of each method on different types of data. While the discriminative methods excel at capturing the underlying complexities of the sentiment in an actual customer’s review, generative methods do not and rather perform well only in artificially generated data, failing to capture nuances present in real customer sentiments.

Overall, this study highlights the difference in performance between the two methods and emphasizes the significance of selecting an appropriate model for the task at hand. While generative models may excel in certain scenarios, such as synthetic data generation, discriminative models prove to be more reliable when dealing with real-world sentiment analysis tasks.

## Citations

- Justifying recommendations using distantly-labeled reviews and fined-grained aspects
  - Jianmo Ni, Jiacheng Li, Julian McAuley
  - Empirical Methods in Natural Language Processing (EMNLP), 2019
  - [Dataset Source](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
