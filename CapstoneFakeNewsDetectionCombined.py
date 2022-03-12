"""
Class:  MIS 581 - Module 8 - Option 1 - Portfolio Project - Diana Stubbe - Dr. den Heijer - 03/13/2022
Topic:  Option #1:  Capstone Project—Final Report and Slide Presentation: U.S. Organization

Source:
Siddhartha, M. (2020, August 10). Fake news detection using NLP and machine learning in Python — Wisdom ML. Medium.
https://medium.com/@sid321axn/fake-news-detection-using-nlp-and-machine-learning-in-python-wisdom-ml-6f548b0691a7

Description: Passive Aggressive Classifier Algorithm and  Multinomial Naive Bayes Classifier techniques are analyzed.
The news data comes from In Search of Truth (ISOT) Fake News Dataset creation Methodology. This dataset was collected
from real-world sources; the truthful articles came from crawling articles from Reuters.com (news website). As for
the fake news articles, they were collected from different sources. The fake news articles were collected from
unreliable websites that were flagged by Politifact (a fact-checking organization in the USA) and Wikipedia.
"""

import warnings
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Removes warnings while running the program
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Change the labels
df.loc[(df['label'] == 1), ['label']] = 'FAKE'
df.loc[(df['label'] == 0), ['label']] = 'TRUE'

# Check for nulls
print(df.isna().sum())
print(df.head())

'''Start of Feature Engineering'''

# Number of words
df['numberOfWords'] = df.text.apply(lambda data: len(data.split()))
# Number of unique words
df['numberUniqueWords'] = df.text.apply(lambda data: len(set(data.split())))
# Number of characters
df['numberOfChar'] = df.text.apply(lambda data: len(data))
# Number of special characters
df["numberSpecialChar"] = df.apply(lambda p: sum(not q.isalpha() for q in p["text"]), axis=1)


# Number of punctuations function
def punct(text):
    return len([w for w in text.split() if w in list(string.punctuation)])


# Plot punctuations
df['numberOfPunct'] = df.text.apply(lambda data: punct(data))
plt.title("Distribution of Punctuation")
plt.show()

# Number of stop words
stopword = stopwords.words('english')


# Number of stop words function
def stop(text):
    return len([w for w in text.split() if w in stopword])


# Plot stop words
df['numberOfStop'] = df.text.apply(lambda data: stop(data))
plt.title("Distribution of stop words")
plt.show()

# Number of most frequent terms
token = nltk.word_tokenize(''.join(df.text))
frequent = nltk.FreqDist(token)
frequent.most_common(15)

# Remove punctuation and stop words from frequent words
for sym in string.punctuation:
    del frequent[sym]
for word in stopword:
    del frequent[word]
frequent.most_common(15)

# Number of words containing the 100 most common words
freq_words = list(dict(frequent.most_common()[:100]).keys())


# Frequency function
def freq(text):
    return len([w for w in text.split() if w in freq_words])


# Plot word frequency
df['NumberOfFreqWords'] = df.text.apply(lambda data: freq(data))
plt.title("Distribution of frequent words")
plt.show()

freq_words = list(dict(frequent.most_common()[:100]).keys())

print(freq_words)


# Frequency function for averaging
def freq(text):
    return len([w for w in text.split() if w in freq_words]) / len(text.split())


# Plot average word frequency
df['avgWordFreq'] = df.text.apply(lambda data: freq(data))
plt.title("Distribution of average word frequency")
plt.show()

'''Start of Exploratory Data Analysis'''

# Plot for Distribution of Fake and True news
sns.set_style("dark")
sns.countplot(df.label)
plt.title("Distribution of Fake and True news")
plt.show()

# Split data sets based on label
df_fake = df[df['label'] == 'FAKE']
df_true = df[df['label'] == 'TRUE']

# Plot for Distribution of the number of characters for Fake news
df_fake['numberOfChar'].plot(bins=50, kind='hist')
plt.title("Distribution of the number of characters for Fake news")
plt.show()

# Describe numberOfChar variable for Fake news
print(df_fake['numberOfChar'].describe())

# Plot for Distribution of the number of characters for True news
df_true['numberOfChar'].plot(bins=50, kind='hist')
plt.title("Distribution of the number of characters for True news")
plt.show()

# Describe numberOfChar variable for True news
print(df_true['numberOfChar'].describe())

# Plot for Distribution of unique words for Fake news
df_fake['numberUniqueWords'].plot(bins=10, kind='hist')
plt.title("Distribution of unique words for Fake news")
plt.show()

# Describe the numberUniqueWords variable for Fake news
print(df_fake['numberUniqueWords'].describe())

# Plot for Distribution of unique words for True news
df_true['numberUniqueWords'].plot(bins=10, kind='hist')
plt.title("Distribution of unique words for True news")
plt.show()

# Describe the numberUniqueWords variable for True news
print(df_true['numberUniqueWords'].describe())

# distribution of fake words
sns.distplot(df_fake['numberOfWords'])
plt.title("Distribution of Words in Fake news")
plt.show()

# distribution of  words in Real news
sns.distplot(df_true['numberOfWords'])
plt.title("Distribution of Words in True news")
plt.show()

# distribution of  special characters in Real news
sns.distplot(df_true['numberSpecialChar'])
plt.title("Distribution of Special chars in True news")
plt.show()

# distribution of  special characters in Fake news
sns.distplot(df_fake['numberSpecialChar'])
plt.title("Distribution of Special chars in Fake news")
plt.show()

# Remove special characters from Fake news as it uses more special characters
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


# Removing unnecessary stop words from news function
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


# Removing unnecessary stop words from news
df['text'] = df['text'].apply(remove_stopwords)


# Plot the confusion matrix
def plot_confusion_matrix(description, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + " - " + description)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# List of special characters
spec_chars = ["-", "!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ".", "/", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "–"]
for char in spec_chars:
    df['text'] = df['text'].str.replace(char, '')

# Define variables for splitting the data into test and training
y = df.label
X = df['text']

# Split the data into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Apply the count vectorizer to the text
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

# Run and plot the Multinomial Naive Bayes Classifier
mnb = MultinomialNB()
mnb.fit(count_train, y_train)
pred = mnb.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("Multinomial Naive Bayes Classifier's Accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'TRUE'])
plot_confusion_matrix("Multinomial Naive Bayes Classifier", cm, classes=['FAKE', 'TRUE'])

# Run and plot the Passive Aggressive Classifier
passive = PassiveAggressiveClassifier()
passive.fit(count_train, y_train)
pred = passive.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("Passive Aggressive Classifier's Accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'TRUE'])
plot_confusion_matrix("Passive Aggressive Classifier", cm, classes=['FAKE', 'TRUE'])
