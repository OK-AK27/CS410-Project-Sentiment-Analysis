
This project compares traditional machine learning approaches and transformer-based models (BERT) for sentiment analysis on the IMDB movie review dataset.

---

# Project Structure

**Sentiment-Analysis-of-Social-Media-Reviews**

│
├ data/                      # Cleaned training & test datasets
│   ├ imdb_train_clean.csv
│   ├ imdb_test_clean.csv
│
├ scripts/                   # All training/evaluation scripts
│   ├ train_classical_models.py
│   ├ train_bert_model.py
│   ├ evaluate_model.py
│   ├ visualize_results.py
│   ├ preprocess_text.py
│
├ results/
│   ├ classical_results.csv
│   ├ metrics.csv
│   ├ confusion_matrix_bert.csv
│   ├ plots/
│       ├ classical_accuracy.png
│       ├ classical_training_time.png
│       ├ classical_vs_bert_accuracy.png
│
├ models/
│   ├ bert_imdb_sentiment/
│      ├ config.json
│      ├ pytorch_model.bin
│
├ README.md
└requirements.txt

# Installation & Setup 

## 1. Clone this repository
git clone <repo-url>
cd Sentiment-Analysis-of-Social-Media-Reviews

## 2. Install dependencies
pip install -r requirements.txt


# Running the Project

## 1. Train Classical Models
python3 scripts/train_classical_models.py

## 2. Train BERT Model
python3 scripts/train_bert_model.py

## 3. Evaluate Models
python3 scripts/evaluate_model.py

## 4. Visualize Results
python3 scripts/visualize_results.py


# Results & Analysis

| Model                   | Accuracy  | Training Time | Notes                       |
| ----------------------- | --------- | ------------- | --------------------------- |
| Multinomial Naive Bayes | 0.843     | 0.01s         | Very fast, lower accuracy   |
| Logistic Regression     | 0.884     | 0.17s         | Best classical model        |
| Linear SVM              | 0.873     | 0.24s         | Good performer              |
| **BERT (base)**         | **0.917** | **12.6 hrs**  | Best accuracy, highest cost |




# Accuracy Comparison – Classical vs. BERT
<img width="779" height="576" alt="image" src="https://github.com/user-attachments/assets/8cfeccac-03bd-4779-a339-83f65361a19e" />

# Sentiment Analysis: Classical NLP vs BERT


# Key Insights:
1. BERT outperformed all classical models by a significant margin
2. Classical models trained quickly but struggled to capture context
3. BERT provides superior language understanding but is computationally expensive

# Confusion Matrix - BERT
[[11013  1487]
 [  581 11919]]
It shows BERT slightly confuses positive sentiment
(example: sarcasm, mixed reviews).

# Conclusion
1. BERT achieves state-of-the-art performance
2. Classical models are lightweight and practical for fast deployment
3. Choice depends on real-world constraints (accuracy vs speed)
