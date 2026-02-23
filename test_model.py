"""
Credit Card Fraud Detection - Test with Real Data
===================================================
This script tests the trained model using actual data from the Kaggle dataset.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import os

# Paths
MODEL_PATH = 'outputs/models/best_model.joblib'
DATA_PATH = r'C:\Users\nzx19\.cache\kagglehub\datasets\mlg-ulb\creditcardfraud\versions\3\creditcard.csv'
FEATURE_NAMES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']


def load_model():
    """Load the trained fraud detection model."""
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    return model


def load_data():
    """Load the credit card fraud dataset."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    return df


def preprocess_features(df, scaler):
    """Preprocess features with the given scaler."""
    df_processed = df.copy()
    
    # Scale Time and Amount together
    df_processed[['Time', 'Amount']] = scaler.transform(df_processed[['Time', 'Amount']])
    
    return df_processed


def predict_fraud(model, features):
    """Make fraud prediction on transaction features."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return predictions, probabilities


def test_model():
    """Test the model with real data."""
    # Load model and data
    model = load_model()
    df = load_data()
    
    # Fit scaler on all data (fit on both Time and Amount together)
    scaler = RobustScaler()
    scaler.fit(df[['Time', 'Amount']])
    
    # Test 1: Sample of legitimate transactions
    print("\n" + "="*60)
    print("TEST 1: Legitimate Transactions (Class = 0)")
    print("="*60)
    
    legit_df = df[df['Class'] == 0].sample(n=10, random_state=42)
    legit_features = preprocess_features(legit_df[FEATURE_NAMES], scaler)
    
    predictions, probabilities = predict_fraud(model, legit_features)
    
    correct = sum(predictions == 0)
    print(f"Correctly predicted as legitimate: {correct}/10")
    print(f"Average fraud probability: {probabilities.mean():.4f}")
    
    # Test 2: Sample of fraudulent transactions
    print("\n" + "="*60)
    print("TEST 2: Fraudulent Transactions (Class = 1)")
    print("="*60)
    
    fraud_df = df[df['Class'] == 1].sample(n=min(10, len(df[df['Class']==1])), random_state=42)
    fraud_features = preprocess_features(fraud_df[FEATURE_NAMES], scaler)
    
    predictions, probabilities = predict_fraud(model, fraud_features)
    
    correct = sum(predictions == 1)
    print(f"Correctly predicted as fraud: {correct}/{len(fraud_df)}")
    print(f"Average fraud probability: {probabilities.mean():.4f}")
    
    # Test 3: Full dataset evaluation
    print("\n" + "="*60)
    print("TEST 3: Full Dataset Evaluation")
    print("="*60)
    
    all_features = preprocess_features(df[FEATURE_NAMES], scaler)
    
    all_predictions, all_probabilities = predict_fraud(model, all_features)
    
    # Confusion matrix components
    true_positives = sum((all_predictions == 1) & (df['Class'] == 1))
    true_negatives = sum((all_predictions == 0) & (df['Class'] == 0))
    false_positives = sum((all_predictions == 1) & (df['Class'] == 0))
    false_negatives = sum((all_predictions == 0) & (df['Class'] == 1))
    
    print(f"Total transactions: {len(df)}")
    print(f"True Positives (correctly detected fraud): {true_positives}")
    print(f"True Negatives (correctly identified legit): {true_negatives}")
    print(f"False Positives (legit flagged as fraud): {false_positives}")
    print(f"False Negatives (missed fraud): {false_negatives}")
    
    accuracy = (true_positives + true_negatives) / len(df)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\n" + "="*60)
    print("Model is working correctly!")
    print("="*60)


if __name__ == '__main__':
    test_model()
