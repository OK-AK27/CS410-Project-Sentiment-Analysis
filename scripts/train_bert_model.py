# scripts/train_bert_model.py

import os
import time
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ---------------- CONFIG ---------------- #

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- DATA LOADING ---------------- #

def load_data():
    """
    Load cleaned IMDB CSVs and ensure there is a 'label' and 'clean_review' column.
    Uses absolute paths so it works no matter where the script is run from.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root (.. from scripts/)
    train_path = os.path.join(base_dir, "data", "imdb_train_clean.csv")
    test_path = os.path.join(base_dir, "data", "imdb_test_clean.csv")

    print("Looking for train CSV at:", train_path)
    print("Looking for test CSV at:", test_path)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    train_df = pd.read_csv(train_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    test_df = pd.read_csv(test_path, encoding="utf-8", engine="python", on_bad_lines="skip")

    # Adjust label column if needed
    if "label" not in train_df.columns:
        if "sentiment" in train_df.columns:
            train_df = train_df.rename(columns={"sentiment": "label"})
            test_df = test_df.rename(columns={"sentiment": "label"})
        else:
            raise ValueError(f"No 'label' or 'sentiment' column in {train_path}")

    if "clean_review" not in train_df.columns:
        raise ValueError(
            f"No 'clean_review' column in {train_path}. "
            f"Available columns: {list(train_df.columns)}"
        )

    return train_df, test_df


# ---------------- DATASET CLASS ---------------- #

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------- TRAIN & EVAL ---------------- #

def train_bert():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_df, test_df = load_data()
    train_texts = train_df["clean_review"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["clean_review"].tolist()
    test_labels = test_df["label"].tolist()

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR)

    # ----- TRAINING LOOP ----- #
    print(f"Training BERT on {len(train_dataset)} examples for {EPOCHS} epochs...")
    start_train = time.time()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    end_train = time.time()
    train_time = end_train - start_train
    print(f"Total training time: {train_time:.2f} seconds")

    # Save model & tokenizer
    save_dir = "models/bert_imdb_sentiment"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved fine-tuned model to {save_dir}")

    # ----- EVALUATION ----- #
    print("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_true = []

    start_infer = time.time()
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"].cpu().numpy()
            all_true.extend(labels)
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    end_infer = time.time()

    infer_time = end_infer - start_infer
    reviews_per_sec = len(all_true) / infer_time if infer_time > 0 else 0.0

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average="binary"
    )
    cm = confusion_matrix(all_true, all_preds)

    print("\n=== BERT Test Metrics ===")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Training time (sec): {train_time:.2f}")
    print(f"Inference time (sec): {infer_time:.2f}")
    print(f"Inference speed: {reviews_per_sec:.2f} reviews/sec")
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n")
    print(classification_report(all_true, all_preds))

    # ----- SAVE PREDICTIONS & CONF MATRIX ----- #

    preds_df = pd.DataFrame({"true": all_true, "pred": all_preds})
    preds_df.to_csv("results/predictions/bert_preds.csv", index=False)

    np.savetxt("results/conf_matrix_bert.csv", cm, fmt="%d", delimiter=",")

    # ----- APPEND METRICS TO results/metrics.csv ----- #

    metrics_row = {
        "Model": "BERT (base)",
        "Accuracy": acc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Training Time (sec)": train_time,
        "Inference Speed (reviews/sec)": reviews_per_sec,
    }

    metrics_path = "results/metrics.csv"
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = metrics_df[metrics_df["Model"] != "BERT (base)"]
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([metrics_row])],
            ignore_index=True,
        )
    else:
        metrics_df = pd.DataFrame([metrics_row])

    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved BERT metrics to {metrics_path}")
    print("Saved predictions to results/predictions/bert_preds.csv")
    print("Saved confusion matrix to results/conf_matrix_bert.csv")


if __name__ == "__main__":
    train_bert()
