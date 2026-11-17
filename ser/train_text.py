"""Train a from-scratch Naive Bayes emotion classifier on utterance transcripts.

Usage:
    python -m ser.train_text [--column AugmentedText] [--test-size 0.2] [--seed 42]
"""
import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ser.data import load_text_csv
from ser.naive_bayes import NaiveBayesClassifier


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--column", default="AugmentedText",
                        choices=["AugmentedText", "Sentences", "text"],
                        help="which transcript column to train on")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = load_text_csv()
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.column], df["Label"], test_size=args.test_size, random_state=args.seed
    )

    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    print(f"Train / test samples: {X_train_counts.shape[0]} / {X_test_counts.shape[0]}")
    print(f"Vocabulary size:      {len(vectorizer.vocabulary_)}")

    clf = NaiveBayesClassifier().fit(X_train_counts, y_train)
    y_pred = clf.predict(X_test_counts)
    print(f"Naive Bayes accuracy ({args.column}): {clf.score(X_test_counts, y_test):.2%}\n")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
