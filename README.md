# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using Scikit-learn and Pandas.

## Overview

This project implements a fraud detection system using the Kaggle Credit Card Fraud Detection dataset. It includes:

- **Data Exploration**: Comprehensive analysis of transaction patterns
- **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost
- **Imbalance Handling**: SMOTE, Random Under-sampling, and combined techniques
- **Model Evaluation**: ROC-AUC, PR-AUC, Confusion Matrix, and more
- **Prediction API**: Scripts to make predictions on new transactions

## Dataset

The dataset contains transactions made by European cardholders in September 2013:

- **Total transactions**: 284,807
- **Fraudulent transactions**: 492 (0.172%)
- **Features**: 30 features (V1-V28 PCA transformed + Time + Amount)

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Jupyter Notebook

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

Run all cells to execute the complete analysis pipeline.

### Option 2: Make Predictions

Use the prediction script to detect fraud in new transactions:

```bash
# Test with real data (10 legit + 10 fraud samples)
python predict.py --test

# Predict custom CSV file
python predict.py --input your_transactions.csv

# Run demo with sample data
python predict.py --demo
```

### Option 3: Test the Model

```bash
python test_model.py
```

This runs comprehensive tests using the actual Kaggle dataset.

## Project Structure

```
credit-card-fraud-detection/
├── data/                       # Dataset directory
├── notebooks/
│   └── fraud_detection.ipynb   # Main analysis notebook
├── outputs/
│   ├── models/                 # Saved models (best_model.joblib)
│   ├── plots/                  # Visualizations
│   └── reports/                # Evaluation reports
├── predict.py                  # Prediction script
├── test_model.py               # Model testing script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Key Results

### Model Performance (Best Model: Random Forest)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 99.94% |
| Precision | 88.61% |
| Recall    | 73.68% |
| F1-Score  | 80.46% |
| ROC-AUC   | 96.44% |
| PR-AUC    | 76.71% |

### Full Dataset Evaluation

- True Positives: 459 (correctly detected fraud)
- True Negatives: 284,258 (correctly identified legit)
- False Positives: 57 (legit flagged as fraud)
- False Negatives: 33 (missed fraud)

## Making Predictions

### Input Format

Your CSV file should contain these 30 columns:

- `Time`: Seconds elapsed since first transaction
- `V1` to `V28`: PCA-transformed features (anonymized)
- `Amount`: Transaction amount

### Example Output

```
Results for 20 transactions:
  Frauds detected: 10
  Legitimate: 10

Details:
  Transaction 1: LEGIT (Actual: LEGIT) (fraud_prob=0.0012)
  Transaction 2: LEGIT (Actual: LEGIT) (fraud_prob=0.0023)
  ...
  Transaction 15: FRAUD (Actual: FRAUD) (fraud_prob=0.9456)
```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- jupyter
- kagglehub

## License

This project is for educational purposes.
