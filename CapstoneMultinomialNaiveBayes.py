"""
Class:  MIS 581 - Module 8 - Option 1 - Portfolio Project - Diana Stubbe - Dr. den Heijer - 03/13/2022
Topic:  Option #1:  Capstone Project—Final Report and Slide Presentation: U.S. Organization

Source:
Gurukul. (2022). Fake News Detection Project in Python with Machine Learning.
https://projectgurukul.org/fake-news-detection-project-python-machine-learning/

Description: Machines are creating an ever-increasing amount of data each second, and there is
concern that this data may be fake. Fake news can put the world at risk. Consider what would
happen if a patient were given the wrong drug based on misleading information.

The project creates a classifier that recognizes whether news is false or not. The model uses
binary classification. TF-IDF Vectorizer is used to preprocess the text data from the dataset. On
the preprocessed text, the Multinomial Naive Bayes technique trains and assesses the model. """

import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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
dataset = pd.concat([true_df, fake_df])

# Change the labels
dataset.loc[(dataset['label'] == 1), ['label']] = 'FAKE'
dataset.loc[(dataset['label'] == 0), ['label']] = 'TRUE'

# View the shape of the data
dataset.shape

# Check for null values
dataset.isnull().sum()

# Check if the dataset is balanced or unbalanced dataset
dataset['label'].value_counts()

# View the shape of the data
true_df.shape
# View the shape of the data
fake_df.shape

# Shuffle or Resample
dataset = dataset.sample(frac=1)

# View top 20 rows
dataset.head(20)

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Stop words
stopwords = stopwords.words('english')

# Only needs to run once to install, then it can be commented out
nltk.download('wordnet')


# Data cleaning
def clean_data(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    token = text.split()
    token = [lemmatizer.lemmatize(word) for word in token if not word in stopwords]
    clean_news = ' '.join(token)

    return clean_news


# Clean the text variable's data
dataset['text'] = dataset['text'].apply(lambda x: clean_data(x))

# Check for nulls
dataset.isnull().sum()

# Apply the TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))

# Partition the X and y labels
X = dataset.iloc[:35000, 0]
y = dataset.iloc[:35000, 1]

# View the labels
X.head()
y.head()

# Split the data into an 80-20 ratio using the train_test_split() function
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert the text data into vectors using the vectorizer
vec_train = vectorizer.fit_transform(train_X)
vec_train = vec_train.toarray()
vec_test = vectorizer.transform(test_X).toarray()
train_data = pd.DataFrame(vec_train, columns=vectorizer.get_feature_names_out())
test_data = pd.DataFrame(vec_test, columns=vectorizer.get_feature_names_out())

# Run the Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(train_data, train_y)

# Predict using testing data
predictions = clf.predict(test_data)

print("Test Classification Report")
print(classification_report(test_y, predictions))

# Predict using training data
predictions_train = clf.predict(train_data)

print("Training Classification Report")
print(classification_report(train_y, predictions_train))

print("Test Data Accuracy Score: ", accuracy_score(test_y, predictions))
print("Training Data Accuracy Score: ", accuracy_score(train_y, predictions_train))
