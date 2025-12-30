
# Interpretable Machine Learning: SHAP-Based Customer Churn Prediction

This project applies a real-world end-to-end machine learning workflow to predict customer churn and explain model decisions using SHAP (SHapley Additive Explanations).
The goal is not only to build an accurate model but also to translate complex predictions into clear, actionable business insights.

## 📌 Project Overview
Customer churn represents a major financial challenge for subscription-based businesses such as telecom, banking, and SaaS. While many ML models can predict churn, organizations increasingly require transparent and interpretable explanations.

This project solves both:
1. Build a strong predictive model (XGBoost)
2. Explain the model globally using SHAP summary plots
3. Explain predictions for individual customers using SHAP force plots
4. Translate model behavior into actionable insights

## 🔧 Technologies Used
- Python 3
- Pandas, NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Matplotlib

## 🧠 Model Development
### 1. Data Preprocessing
- Dropped identifiers (RowNumber, CustomerId, Surname)
- Label-encoded categorical columns
- Train-test split (80/20)

### 2. Model Training
XGBoostClassifier with:
- 500 estimators
- depth = 5
- learning_rate = 0.05
- subsample = 0.8

### 3. Cross-Validation
Mean CV AUC ≈ **0.858**

## 📊 Model Performance
| Metric      | Score |
|-------------|--------|
| Accuracy    | 0.87   |
| F1-score    | 0.61   |
| ROC-AUC     | 0.865  |

## 🔍 SHAP Interpretations
### Global Importance
Top churn drivers:
- Age
- Balance
- Number of Products
- Active Membership
- Credit Score

### Local Explanations
Three customer SHAP force plots:
- customer0.png (Low Risk)
- customer1.png (Medium/Borderline Risk)
- customer2.png (High Risk)

## 📁 Project Structure
project/
│ README.md
│ churn_model.ipynb
│ Churn_Modelling.csv
│ customer0.png
│ customer1.png
│ customer2.png

## 🏁 Conclusion
This project demonstrates strong churn prediction performance and clear interpretability suitable for business and regulatory use.

Created by Sivanandham (sir).
