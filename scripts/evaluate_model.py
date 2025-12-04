import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# Load test data
df_test = pd.read_csv("data/imdb_test.csv")

# Load tokenizer and model
model_path = "results/bert-model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Tokenize test data
test_encodings = tokenizer(df_test["text"].tolist(), truncation=True, padding=True, return_tensors="pt")
test_labels = df_test["label"].tolist()

# Wrap in Hugging Face Dataset
test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

# Setup Trainer
trainer = Trainer(model=model)

# Predict
outputs = trainer.predict(test_dataset)
probs = torch.softmax(torch.tensor(outputs.predictions), dim=1)
preds = torch.argmax(probs, axis=1).numpy()

# Evaluation
print("\nClassification Report:")
print(classification_report(test_labels, preds, digits=4))

# Confusion Matrix
cm = confusion_matrix(test_labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# ROC AUC Curve (binary classification)
fpr, tpr, _ = roc_curve(test_labels, probs[:, 1])
auc_score = roc_auc_score(test_labels, probs[:, 1])
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()

# Export predictions
df_test["true_label"] = test_labels
df_test["predicted_label"] = preds
df_test.to_csv("results/predictions.csv", index=False)

# Save misclassified examples
misclassified = df_test[df_test["true_label"] != df_test["predicted_label"]]
misclassified.to_csv("results/misclassified.csv", index=False)

print(f"\n✅ Predictions saved to results/predictions.csv")
print(f"❌ Misclassified examples saved to results/misclassified.csv")
print(f"📊 Confusion matrix saved to results/confusion_matrix.png")
print(f"📈 ROC curve saved to results/roc_curve.png")

with open("results/metrics_summary.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"AUC Score: {auc:.4f}\n")

