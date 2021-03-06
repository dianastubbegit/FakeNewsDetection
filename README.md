# FakeNewsDetection
Fake News Detection

Almost everyone gets news from several online news sources in today's world. As the usage of social media platforms such as Facebook, Twitter, and others has grown, information propagates to millions of individuals in a short period.

False news can have far-reaching consequences, such as establishing skewed perceptions to influence election outcomes for politicians. Furthermore, spammers profit from click-bait advertisements by using compelling news headlines. 

With Artificial Intelligence, Natural Language Processing, and Machine Learning techniques, it is possible to categorize various news articles available online as fake or real news.

# DataCleaning
Data Cleaning

Punctuation can help grasp a phrase by providing grammatical context. However, to model text data, it is necessary to eliminate all special characters because the vectorizer only counts the number of words, not the context.

Tokenization separates text into components such as sentences or words. It provides previously unstructured text structure.

Stopwords are frequent words used in almost any writing. The cleaning process eliminates them since they don't provide much information about our data.

# Models
Modeling Techniques

Two modeling techniques are analyzed.

A typical Bayesian learning methodology in natural language is the Multinomial Naive Bayes method. The algorithm calculates a text's tag, such as a newspaper article. Then, it estimates each tag's likelihood for a given sample and returns the tag with the highest probability. The accuracy is close to 95%.

The Passive-Aggressive Classifier is a machine learning technique that responds passively to correct classifications and aggressively to any miscalculations, allowing it to self-update and modify regularly. The accuracy is close to 99%.

# FileDescriptions
File Descriptions

   - Fake.csv: Kaggle sourced comma-delimited file that contains data collected from unreliable websites that were flagged by Politifact
     https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset  (Only 100 rows loaded due to size limitations)

   - True.csv: Kaggle sourced comma-delimited file that contains data collected from real-world sources from crawling articles from Reuters
     https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset (Only 100 rows loaded due to size limitations)
     
   - CombinedNews.csv: Kaggle sourced comma-delimited file that contains data from Fake.csv and True.csv (Only 200 rows loaded due to size limitations)
     https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

   - CapstoneChiSquareHypothesis.py: python code to analyze the null hypothesis

   - CapstoneMultinomialNaiveBayes.py: python code to analyze the Multinomial Naive Bayes Classifier

   - CapstonePassiveAggressive.py: python code to analyze the Passive-Aggressive Classifier

   - CapstoneFakeNewsDetectionCombined.py: python code to analyze the two Classifiers concurrently

   - Module8PowerBI.pbix: Power BI Report that summarizes the data in the Fake.csv and True.csv files (Only 200 rows available in the report due to size limitations)
