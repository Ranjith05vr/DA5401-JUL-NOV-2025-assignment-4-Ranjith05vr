# DA5401 A4: GMM-Based Synthetic Sampling for Imbalanced Data

## Objective
This assignment explores **Gaussian Mixture Model (GMM)-based synthetic sampling** to handle imbalanced data in the **Credit Card Fraud Detection** dataset.  
We compare three approaches:
1. **Baseline Model** – Logistic Regression on the original imbalanced dataset.  
2. **GMM Oversampling** – Minority class is oversampled using GMM until it matches the majority.  
3. **Undersampling + GMM** – Majority is undersampled to a suitable size, then minority is oversampled with GMM to match.  

The results are compared using **Precision, Recall, F1-score, and ROC AUC**.

---

## Dataset
- **Source**: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- File used: `creditcard.csv`  
- Already PCA-transformed (`V1 … V28`), with two additional features (`Time` and `Amount`).  
- Highly imbalanced: only ~0.17% of transactions are fraud.

---

## Experiments
### Part A: Baseline Model
- Logistic Regression trained on the original imbalanced training set.  
- Metrics show **poor Recall** due to imbalance.  
- Accuracy is misleading → Precision/Recall/F1 are emphasized.

### Part B: GMM Oversampling
- Fit a **Gaussian Mixture Model** to minority class (fraud) samples.  
- Generate synthetic minority samples until minority = majority.  
- Retrain Logistic Regression on balanced data.  
- Performance improves significantly, especially Recall.

### Part C: Undersample Majority + GMM Oversampling
- Apply **clustering-based undersampling** to reduce majority to ~2× minority.  
- Apply GMM-based oversampling to minority to balance.  
- Retrain Logistic Regression.  
- Balanced performance with reduced computation cost.

### Part D: Comparison
- Summary table and bar chart comparing all three setups.  
- GMM oversampling improves detection of minority class.  
- Undersampling + GMM trades some precision for computational efficiency.

---

## Evaluation Metrics
- **Precision** – correctness of predicted frauds.  
- **Recall** – ability to detect frauds (important in financial risk).  
- **F1-score** – balance between precision and recall.  
- **ROC AUC** – overall ranking ability.

---

## How to Run
1. Open **Google Colab**.  
2. Upload `creditcard.csv` into `/content/`.  
3. Open the notebook `DA5401_Assignment_4.ipynb`.  
4. Run all cells in order.  
5. Final section shows comparison plots and results.

---

## Recommendation
- GMM-based oversampling is more effective than the baseline.  
- **Full oversampling** (Part B) yields best Recall → useful when catching fraud is critical.  
- **Undersampling + GMM** (Part C) is efficient and avoids extreme dataset size.  
- For real-world fraud detection, **Part C is recommended**: good Recall, manageable dataset, avoids overfitting.

---

## Files
- `DA5401_Assignment_4.ipynb` – Colab notebook (main report).  
- `README.md` – project overview and instructions.  
- `creditcard.csv` – dataset (must be downloaded separately from Kaggle).  

---


