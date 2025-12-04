# Sentiment Analysis: Classical NLP vs BERT

This project compares traditional machine learning approaches and transformer-based models (BERT) for sentiment analysis on the IMDB movie review dataset.

---

## 📁 Project Structure

project-root/
│
├── data/ # Raw and cleaned IMDB datasets
├── results/ # Model outputs, confusion matrices, evaluation CSVs
├── models/ # (Optional) Trained model weights
├── notebooks/ # Jupyter notebooks (if any)
├── scripts/ # Python scripts for all phases
│ ├── download_dataset.py
│ ├── preprocess_text.py
│ ├── load_imdb_data.py
│ ├── train_classical_model.py
│ ├── train_bert_model.py
│ ├── evaluate_model.py
│ └── plot_results.py
├── requirements.txt
└── README.md