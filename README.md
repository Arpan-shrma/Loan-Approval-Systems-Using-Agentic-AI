# 🧠 Loan Default Prediction with Attention-based Transformer Models

This project explores a hybrid deep learning approach for **loan default prediction** using both **structured financial features** and **unstructured borrower descriptions**. We integrate **DistilBERT** for text embeddings, apply **attention visualization**, and combine it with custom tabular neural networks for robust classification and interpretability.

---

## 📁 Notebooks

### 1. `data_prepration.ipynb` – Data Preprocessing

This notebook prepares the LendingClub dataset for modeling, including:
- **Loading and exploring the dataset** with statistical summaries and visualizations.
- **Outlier handling** (retained due to their real-world relevance).
- **Feature engineering**:  
  - Region and FICO-based bins  
  - Financial ratios (debt-to-income, loan-to-income, etc.)  
- **Text cleaning**: spell correction, sentiment tagging, and standardization.
- **Exploratory Data Analysis**:
  - KDE plots and pairplots for numeric distributions
  - Distribution analysis by default status
  - Text column frequency and sentiment distributions

---

### 2. `Modelling_and_attention_analysis.ipynb` – Hybrid Modeling and Interpretability

This notebook contains the core ML pipeline:
- **Hybrid Model Architecture**:
  - DistilBERT for borrower narratives
  - Dense neural nets for tabular data
  - Combined into a unified hybrid model (`EfficientHybridModel`)
- **Training Strategies**:
  - Mixed-precision training with gradient accumulation
  - Stratified sampling and class balancing
  - Randomized hyperparameter search

#### 📈 Model Performance (Test Set)
| Metric              | Score   | Improvement |
|---------------------|---------|-------------|
| ROC-AUC             | 0.7755  | +16.73%     |
| Balanced Accuracy   | 0.7063  | +14.16%     |
| F1 Score            | 0.6828  | -           |

---

## 🔍 Attention Analysis & Interpretability

- Extracted attention weights from DistilBERT to understand which tokens were most influential.
- **Visualizations**:
  - Attention heatmaps (Default vs Non-default)
  - CLS token distribution
  - Token-level saliency and semantic groupings
  - Confusion matrix and structured token-token maps

---

## 🛠 Tech Stack

- Python, PyTorch, HuggingFace Transformers
- Pandas, Matplotlib, Seaborn
- Sklearn, SpellChecker
- Stratified sampling, mixed-precision training

---

## 📌 Key Takeaways

- Combining financial indicators with semantic text analysis significantly improves classification accuracy.
- Attention visualization provides transparency, identifying meaningful terms like “consolidation” and “urgent” in predicting loan default.
- The final model is interpretable, generalizable, and deployable for real-world financial decision-making systems.

---

## 📂 Directory Structure
```
├── data_prepration.ipynb 
├── Modelling_and_attention_analysis.ipynb 
|── README.md 
└── data/ 
    └── lending_club_loan_data.csv
```
---

## ✨ Future Work
- Expand to multilingual borrower narratives
- Deploy model via REST API for real-time loan screening
