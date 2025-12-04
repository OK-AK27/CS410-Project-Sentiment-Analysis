import pandas as pd
from datasets import Dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load cleaned CSVs
train_df = pd.read_csv("data/imdb_train_clean.csv")
test_df = pd.read_csv("data/imdb_test_clean.csv")

# 2. Convert to HuggingFace Dataset format
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# 3. Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["clean_review"], truncation=True, padding="max_length", max_length=512)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# 4. Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 5. Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="results/bert_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2
)

# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train
trainer.train()

# 9. Save model and tokenizer
trainer.save_model("models/bert_imdb_sentiment")
tokenizer.save_pretrained("models/bert_imdb_sentiment")
