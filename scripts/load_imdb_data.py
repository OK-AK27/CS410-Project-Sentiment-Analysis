import os
import pandas as pd
import glob

def load_reviews_from_folder(folder_path, label):
    texts = []
    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as file:
            texts.append(file.read())
    return pd.DataFrame({"review": texts, "label": label})

def main():
    base_path = "data/aclImdb"

    # Load and label training data
    train_pos = load_reviews_from_folder(os.path.join(base_path, "train", "pos"), 1)
    train_neg = load_reviews_from_folder(os.path.join(base_path, "train", "neg"), 0)
    train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

    # Load and label test data
    test_pos = load_reviews_from_folder(os.path.join(base_path, "test", "pos"), 1)
    test_neg = load_reviews_from_folder(os.path.join(base_path, "test", "neg"), 0)
    test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

    # Save to CSV
    train_df.to_csv("data/imdb_train.csv", index=False)
    test_df.to_csv("data/imdb_test.csv", index=False)
    print("✅ Saved imdb_train.csv and imdb_test.csv")

if __name__ == "__main__":
    main()
