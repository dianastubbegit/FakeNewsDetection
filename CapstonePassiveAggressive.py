"""
Class:  MIS 581 - Module 8 - Option 1 - Portfolio Project - Diana Stubbe - Dr. den Heijer - 03/13/2022
Topic:  Option #1:  Capstone Project—Final Report and Slide Presentation: U.S. Organization

Source:
Dounis, F. (2020). Detecting Fake News With Python And Machine Learning.
https://medium.com/swlh/detecting-fake-news-with-python-and-machine-learning-f78421d29a06

Description: Passive Aggressive Classifier Algorithms that do online learning are known as passive-aggressive
algorithms (with for example twitter data). They have the characteristic of remaining passive while coping with a
correctly categorized outcome and becoming aggressive when a miscalculation occurs, thereby constantly self-updating
and modifying.

By using different iterations of the code and datasets, maximum accuracy can be reached, with an average of 98% to
99%. It is more accurate than the Multinomial Naive Bayes model with a median accuracy of 95%. """

import pandas as pd
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the comma-delimited files to a data frame
path = path = "./"
true_df = pd.read_csv(path + 'True.csv')
fake_df = pd.read_csv(path + 'Fake.csv')

# Add the label column to track true and fake news
true_df['label'] = 0
fake_df['label'] = 1

# Display the data frames for true and fake news
true_df.head()
fake_df.head()

# Keep relevant columns only
true_df = true_df[['text', 'label']]
fake_df = fake_df[['text', 'label']]

# Concatenate the two datasets
df = pd.concat([true_df, fake_df])

# View dataset shape
print(df.shape)

# View top rows
print(df.head())

# Change the labels
df.loc[(df['label'] == 1), ['label']] = 'FAKE'
df.loc[(df['label'] == 0), ['label']] = 'TRUE'

# Isolate the labels
labels = df.label
labels.head()

# Number of words used
df['numberWords'] = df.title.apply(lambda x: len(x.split()))

# Create two data frames
dfFake = df[df['label'] == 'FAKE']
dfTrue = df[df['label'] == 'TRUE']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'].values.astype('str'), labels, test_size=0.2,
                                                    random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the train set and transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize the PassiveAggressiveClassifier and fit the training sets
pa_classifier = PassiveAggressiveClassifier(max_iter=50)
pa_classifier.fit(tfidf_train, y_train)

# Predict and calculate accuracy for testing dataset
y_pred = pa_classifier.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print("Classification Report")
print(classification_report(y_test, y_pred))
print(f'Accuracy Score: {round(score * 100, 2)}%')
