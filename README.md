# 🛒 Purchase Prediction — Support Vector Machine (SVM)

A machine learning project that predicts whether a user will **purchase a product** based on their Age and Estimated Salary using Support Vector Machine with RBF kernel.

---

## 📌 Problem Statement

Given a user's Age and Salary from a Social Network Ad campaign, predict:
- `1` → **Will Purchase** ✅
- `0` → **Will Not Purchase** ❌

---

## 📊 Dataset

| Detail | Info |
|--------|------|
| Source | Social Network Ads Dataset |
| Rows | 400 users |
| Features Used | Age, EstimatedSalary |
| Target | `Purchased` (0 or 1) |
| Missing Values | None |
| Class Balance | 257 Not Purchased, 143 Purchased |

**Dropped Columns:**
```
User ID → just serial number, no value
Gender  → 0.1% difference between male/female → no impact
```

---

## 🤖 Model Used

**Support Vector Machine (SVC) with RBF Kernel**

SVM finds the **best hyperplane** that separates buyers from non-buyers with the **maximum margin**.

```
Non-buyers 🔴     |     🔵 Buyers
                  |
       ← margin → | ← margin →
                  |
            Hyperplane (decision boundary)
```

RBF Kernel transforms the data to find **curved boundaries** when data is not linearly separable.

---

## ⚙️ Steps Followed

1. Load dataset and explore
2. Drop `User ID` and `Gender` columns
3. Split X (Age, Salary) and y (Purchased)
4. Train test split — 80% train, 20% test
5. Apply **StandardScaler** — scaling is critical for SVM!
6. Train **two SVM models** and compare:
   - Model 1: Default `SVC()` — RBF kernel, C=1
   - Model 2: `SVC(C=0.1, gamma=1000)` — bad parameters for comparison
7. Visualize decision boundary
8. Evaluate with confusion matrix and classification report

---

## 📈 Results

| Model | C | Gamma | Accuracy |
|-------|---|-------|----------|
| SVM 1 (Default) | 1.0 | scale | **90%** ✅ |

**Confusion Matrix (SVM 1 — 90%):**
```
                 Predicted
              Not Bought  Bought
Actual Not  [    49         4  ]   → 92% precision
       Bought[    4        23  ]   → 85% precision
```

**Support Vectors:**
```
Class 0 (Not Purchased): 42 support vectors
Class 1 (Purchased):     43 support vectors
Total:                   85 out of 320 training points
Only 26% of points define the entire model! ✅
```

---

## 📉 Visualizations

### 1. Class Distribution
Bar chart showing 257 non-buyers vs 143 buyers

### 2. Age vs Salary Scatter Plot
Original data colored by purchase decision

### 3. SVM Decision Boundary ⭐
Curved RBF boundary separating buyers from non-buyers — the most important visualization showing how SVM draws the decision line

### 4. Confusion Matrix Heatmap
Color-coded matrix showing correct vs wrong predictions

### 5. Model Comparison Bar Chart
SVM 1 (90%) vs SVM 2 (66%) side by side

---

## 🛠️ Libraries Used

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/charan-d11/Social-Network-Ads-Prediction.git


# Open notebook
jupyter notebook SVM_Social_Network_Ad.ipynb
```

---

## 💡 Key Learnings

- SVM finds the **maximum margin hyperplane** between classes
- **Feature scaling is mandatory** for SVM (distance-based algorithm)
- `kernel='rbf'` handles non-linear boundaries using the kernel trick
- High `gamma` (1000) = severe overfitting — small influence area per point
- Low `C` (0.1) = underfitting — too many misclassifications allowed
- Best settings: `C=1, gamma='scale'` → 90% accuracy ✅
- Support vectors are the **only points that define the model** — rest are irrelevant
- Only 2 features needed for a strong 90% model!

---

## 🔑 Key Parameters

| Parameter | Value Used | Effect |
|-----------|-----------|--------|
| `kernel` | `rbf` | Curved decision boundary |
| `C` | `1.0` | Balanced margin control |
| `gamma` | `scale` | Auto-calculated, prevents overfitting |

---

## 📁 Files

```
04_SVM_Social_Network_Ads/
├── NoteBook
    |__Social_Network_Ad.ipynb
    |
        
├── data
    |___Social_Network_Ads.csv
    |
        
└── README.md
```

---

**AUTHOR:**
  **Durga Charan Mallick**