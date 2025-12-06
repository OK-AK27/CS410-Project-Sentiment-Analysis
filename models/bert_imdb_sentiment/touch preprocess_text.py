import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)      # remove HTML line breaks
    text = re.sub(r"[^a-z\s]", "", text)        # remove punctuation, numbers
    text = re.sub(r"\s+", " ", text)            # normalize whitespace
    return text.strip()

def main():
    # Load data
    train_df = pd.read_csv("data/imdb_train.csv")
    test_df = pd.read_csv("data/imdb_test.csv")

    # Clean reviews
    train_df["clean_review"] = train_df["review"].apply(clean_text)
    test_df["clean_review"] = test_df["review"].apply(clean_text)

    # Save processed data
    train_df.to_csv("data/imdb_train_clean.csv", index=False)
    test_df.to_csv("data/imdb_test_clean.csv", index=False)
    print("Saved cleaned train/test data.")

if __name__ == "__main__":
    main()
