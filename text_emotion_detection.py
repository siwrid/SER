import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
df = pd.read_csv('iemocap.csv')

# We'll use CountVectorizer for feature extraction
vectorizer = CountVectorizer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['AugmentedText'], df['Label'], test_size=0.2, random_state=42)

# Transform the text data into numerical vectors
X_train_counts = vectorizer.fit_transform(X_train).toarray()
X_test_counts = vectorizer.transform(X_test).toarray()
feature_names = vectorizer.get_feature_names_out()

# Naive Bayes Classifier Implementation
class NaiveBayesClassifier:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        # Count the number of occurrences of each class in the target vector (log prior)
        _, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))
        
        # Calculate the total count of all features for each class
        self.classes_ = np.unique(y)
        class_feature_counts = np.zeros((len(self.classes_), X.shape[1]))
        for idx, cls in enumerate(self.classes_):
            class_feature_counts[idx, :] = X[y == cls].sum(axis=0)
        
        # Add 1 to all feature counts to avoid division by zero (Laplace smoothing)
        smoothed_counts = class_feature_counts + 1
        
        # Convert the count to log probabilities
        self.feature_log_prob_ = np.log(smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True))

    def predict_log_proba(self, X):
        # Calculate the log probability of each class for each sample
        return (X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X):
        # Get the class with the highest probability
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

# Create and train the Naive Bayes classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train_counts, y_train)

# Predict emotions on the test set
y_pred = nb_classifier.predict(X_test_counts)

# Calculate accuracy manually
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')
