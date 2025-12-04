from sklearn.metrics import accuracy_score
import csv

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import time

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, name, writer):
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"Training time: {training_time:.2f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(cm)

    # Save metrics
    writer.writerow([name, f"{training_time:.2f}", f"{accuracy:.4f}"])

    # Plot and save confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.savefig(f"results/conf_matrix_{name.replace(' ', '_').lower()}.png")
    plt.close()


    # Save to CSV
    writer.writerow([name, f"{training_time:.2f}", f"{accuracy:.4f}"])


def main():
    # Load data
    train_df = pd.read_csv("data/imdb_train_clean.csv")
    test_df = pd.read_csv("data/imdb_test_clean.csv")

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["clean_review"])
    X_test = vectorizer.transform(test_df["clean_review"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    models = [
        (MultinomialNB(), "Multinomial Naive Bayes"),
        (LogisticRegression(max_iter=200), "Logistic Regression"),
        (LinearSVC(), "Linear SVM")
    ]

    with open("results/classical_results.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Training Time (s)", "Accuracy"])

        for model, name in models:
            train_and_evaluate_model(X_train, y_train, X_test, y_test, model, name, writer)


if __name__ == "__main__":
    main()
