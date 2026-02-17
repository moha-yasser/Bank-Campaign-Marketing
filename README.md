# Bank-Marketing-Campaign
An end-to-end banking marketing prediction pipeline using SQL for deduplication and feature engineering, and Python (Gradient Boosting/LightGBM) to handle imbalanced data and predict term deposit subscriptions.

## 📊 Data Source
The dataset used in this project is the Bank Marketing (UCI) dataset, sourced from Kaggle.
- Dataset Link: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
- Description: This dataset contains 41,188 rows representing direct marketing campaigns (phone calls) of a Portuguese banking institution.

## 📌 Project Overview
This project predicts whether a client will subscribe to a bank term deposit based on a Portuguese marketing campaign. The workflow is split into two distinct phases:
- SQL-based Data Engineering: Focusing on data integrity, cleaning, and creating new domain-specific metrics.
- Python-based Modeling: Utilizing automated pipelines and specialized techniques to handle class imbalance and feature selection.

## 🛠️ Tech Stack
- Database: MySQL Workbench
- Environment: Visual Studio Code (Jupyter notebook)
- Libraries: SQLAlchemy, Pandas, Scikit-Learn, LightGBM

## 🗄️ Phase 1: Data Engineering (MySQL Workbench)
I performed the initial heavy lifting in SQL to ensure the dataset was clean and augmented with meaningful features before any modeling took place.

### 🛡️ Data Preservation & Staging
Multi-Stage Backups: I implemented a staging workflow by creating bank_additional_staging and bank_additional_staging2 to perform transformations while keeping the original source data untouched.

### 🧹 Data Cleaning & Quality Audit
- Deduplication: Identified and removed duplicate records by partitioning over 20 unique features using ROW_NUMBER().
- Standardization: Conducted a quality audit using SELECT DISTINCT and LENGTH() to find spelling inconsistencies.
- Uniform Casing: Standardized all categorical columns (Job, Marital, Education, etc.) to lowercase using LOWER() to ensure consistent grouping.

### 🧪 Feature Engineering
I engineered several composite features to provide stronger signals for the models:
- financial_stress: A sum of credit default, housing, and personal loan status (mapped to 1 if 'yes') to measure total debt burden.
- economic_index: A composite score of employment variation rate, consumer confidence, and the euribor 3-month rate.
- contact_intensity: Sum of campaign and previous contacts to track total engagement.
- unknown_count: A flag identifying profiles with missing financial data in the source.

## 🐍 Phase 2: Python Modeling (VS Code)
The modeling phase focused on reproducibility and addressing the significant class imbalance in the target variable.

### ⚖️ Handling Imbalanced Data
The target variable was highly imbalanced (32,717 'No' vs 2,262 'Yes'). To ensure the model learned to identify the minority class effectively:
- Class Weights: Utilized class_weight='balanced' in the models to penalize misclassifications of the minority class more heavily.
- Scoring Metric: Used F1-Score as the primary metric for permutation_importance to ensure feature selection favored a balance between Precision and Recall rather than simple Accuracy.

- Target Encoding: Utilized Label Encoder on the target variable 'y', fitting the encoder on the training set and transforming the test set to maintain strict evaluation standards.

### 🤖 Machine Learning Pipelines
I built two separate pipelines to automate preprocessing and prevent data leakage:

- Gradient Boosting (pipe_gb)
Stages: One-Hot Encoding for temporal features (month, day_of_week) followed by the Gradient Boosting classifier.
Optimization: Used permutation_importance to isolate a specific list of important_features and re-fitted the model for better generalization.

- LightGBM (pipe_lgb)
Categorical Handling: Leveraged LightGBM's native capability to process categorical features. I converted the relevant columns to the category Dtype in a loop before training.
Optimization: Applied the same permutation importance logic to refine the feature set and re-fitted the model.

## 📊 Final Model Comparison
After hyperparameter tuning and feature selection, both the Gradient Boosting and LightGBM models were evaluated. Because the dataset is heavily imbalanced, I focused on the Recall and F1-Score for the minority class (subscribers) to measure success.

| Model | Train Accuracy | Test Accuracy | Precision (Yes) | Recall (Yes) | F1-Score (Yes) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | 73.57% | 72.97% | 0.13 | **0.53** | 0.21 |
| **LightGBM** | **75.22%** | **74.89%** | **0.14** | 0.51 | 0.21 |

While LightGBM achieved higher overall accuracy, Gradient Boosting was selected as the preferred model for identifying potential subscribers due to its superior Recall (0.53), which minimizes the number of missed opportunities in the marketing campaign.

### Detailed Classification Reports

- Gradient Boosting
```bash
precision    recall  f1-score   support

           0       0.96      0.74      0.84      6528
           1       0.13      0.53      0.21       468

    accuracy                           0.73      6996
```

- LightGBM
```bash
precision    recall  f1-score   support

           0       0.96      0.77      0.85      6528
           1       0.14      0.51      0.21       468

    accuracy                           0.75      6996
```

## 📁 Project Structure

```bash
├── data/
│   ├── raw/
│   │   └── bank_marketing.csv                 # Original, unedited Kaggle data
│   └── processed/
│       └── bank_marketing_cleaned.csv         # Post-SQL cleaning & feature engineering
├── sql/
│   └── bank_marketing_etl.sql                 # MySQL cleaning & engineering script
├── notebooks/
│   └── bank_marketing_ml_pipeline.py          # Python pipeline, tuning, & evaluation
├── README.md                                  # Documentation
```

