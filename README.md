# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using Scikit-learn and Pandas.

## Overview

This project implements a fraud detection system using the Kaggle Credit Card Fraud Detection dataset. It includes:

- **Data Exploration**: Comprehensive analysis of transaction patterns
- **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost
- **Imbalance Handling**: SMOTE, Random Under-sampling, and combined techniques
- **Model Evaluation**: ROC-AUC, PR-AUC, Confusion Matrix, and more

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

1. Open the Jupyter Notebook:

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

2. Run all cells to execute the complete analysis pipeline.

## Project Structure

```
credit-card-fraud-detection/
├── data/                       # Dataset directory
├── notebooks/
│   └── fraud_detection.ipynb   # Main analysis notebook
├── outputs/
│   ├── models/                 # Saved models
│   ├── plots/                  # Visualizations
│   └── reports/                # Evaluation reports
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Key Results

The notebook provides:

- Class distribution analysis
- Feature importance rankings
- Model comparison table
- Best model recommendation
- Optimal threshold analysis

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- jupyter

## License

This project is for educational purposes.
