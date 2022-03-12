"""
Class:  MIS 581 - Module 8 - Option 1 - Portfolio Project - Diana Stubbe - Dr. den Heijer - 03/13/2022
Topic:  Option #1:  Capstone Project—Hypothesis Testing

Source:
Bonaros, B. (2021, November 1). A complete guide of how to choose and apply the right statistical test in Python.
Medium.
https://towardsdatascience.com/a-complete-guide-of-how-to-choose-and-apply-
the-right-statistical-test-in-python-5fcaf5fb9351
"""

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.stats import chi2_contingency

# Load the comma-delimited files to a data frame
path = "./"
true_df = pd.read_csv(path + 'True.csv')
fake_df = pd.read_csv(path + 'Fake.csv')

# Add the label column to track true and fake news
true_df['label'] = 'TRUE'
fake_df['label'] = 'FAKE'

# Create data frames for true and fake news with a smaller sample
true_df = true_df[['text', 'label']].sample(n=5000)
fake_df = fake_df[['text', 'label']].sample(n=5000)

# Concatenate the two datasets
dataset = pd.concat([true_df, fake_df])

# Check null values
dataset.isnull().sum()  # no null values

# Check the balance of the dataset
dataset['label'].value_counts()

# Check the data frame shapes
true_df.shape
fake_df.shape

# Shuffle the dataset
dataset = dataset.sample(frac=1)

# Apply the lemmatizer
lemmatizer = WordNetLemmatizer()

# Apply stop words
stopwords = stopwords.words('english')


# Data cleaning function
def clean_data(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    token = text.split()  # row.split()
    token = [lemmatizer.lemmatize(word) for word in token if not word in stopwords]
    clean_news = ' '.join(token)  # news

    return clean_news


# Clean the data
dataset['text'] = dataset['text'].apply(lambda x: clean_data(x))

# Check for nulls
print(dataset.isnull().sum())

# Chi-square test of independence.
contingency = pd.crosstab(dataset['text'], dataset['label'])

c, p, dof, expected = chi2_contingency(contingency)

print("Chi-square test of independence p_value: ", round(p, 3))
