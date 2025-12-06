import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

metrics_path = "results/metrics.csv"
df = pd.read_csv(metrics_path)

# --- Accuracy comparison ---
plt.figure(figsize=(8, 6))
plt.bar(df["Model"], df["Accuracy"])
plt.title("Accuracy Comparison – Classical vs BERT")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)

for i, row in df.iterrows():
    plt.text(i, row["Accuracy"] + 0.005, f"{row['Accuracy']:.3f}", ha="center")

plt.tight_layout()
plt.savefig("results/plots/accuracy_all_models.png")
plt.show()

# --- Training time (log scale because BERT is huge) ---
if "Training Time (sec)" in df.columns:
    plt.figure(figsize=(8, 6))
    plt.bar(df["Model"], df["Training Time (sec)"])
    plt.title("Training Time (sec) – Classical vs BERT")
    plt.ylabel("Time (seconds)")
    plt.yscale("log")  # highlights the massive gap

    for i, row in df.iterrows():
        plt.text(
            i,
            row["Training Time (sec)"] * 1.1,
            f"{row['Training Time (sec)']:.2f}",
            ha="center",
            rotation=90,
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("results/plots/training_time_all_models.png")
    plt.show()

    import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_conf_matrix_from_csv(csv_path, model_name):
    cm = np.loadtxt(csv_path, delimiter=",", dtype=int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neg", "Pos"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    out_path = f"results/plots/conf_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path)
    plt.show()

# Call once for BERT:
# plot_conf_matrix_from_csv("results/conf_matrix_bert.csv", "BERT (base)")

