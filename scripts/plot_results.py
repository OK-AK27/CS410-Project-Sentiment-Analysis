import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/classical_results.csv")

# Bar Plot - Accuracy
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Accuracy"].astype(float))
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0.7, 1.0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("results/classical_accuracy.png")
plt.show()
