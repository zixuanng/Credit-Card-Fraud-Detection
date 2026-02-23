"""
Credit Card Fraud Detection - Prediction Script
=================================================
This script loads the trained model and makes predictions on new transactions.

Usage:
    python predict.py                    # Run with sample test data
    python predict.py --input data.csv   # Run with custom CSV file
    python predict.py --test             # Test with real Kaggle data

The model expects 30 features:
    - Time: Seconds elapsed since first transaction
    - V1 to V28: PCA-transformed features (anonymized)
    - Amount: Transaction amount
"""

import joblib
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import RobustScaler
import os


# Model and paths
MODEL_PATH = 'outputs/models/best_model.joblib'
DATA_PATH = r'C:\Users\nzx19\.cache\kagglehub\datasets\mlg-ulb\creditcardfraud\versions\3\creditcard.csv'
FEATURE_NAMES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']


def load_model():
    """Load the trained fraud detection model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run the Jupyter notebook first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    return model


def create_and_fit_scaler():
    """Create and fit scaler on the original dataset for consistent preprocessing."""
    if os.path.exists(DATA_PATH):
        # Load data and fit scaler
        df = pd.read_csv(DATA_PATH)
        scaler = RobustScaler()
        scaler.fit(df[['Time', 'Amount']])
        return scaler
    else:
        # Return a new scaler if data not available
        print("Warning: Original dataset not found. Using default scaler.")
        return RobustScaler()


def preprocess_data(df, scaler):
    """
    Preprocess the input data using RobustScaler for Time and Amount.
    
    Parameters:
        df: DataFrame with columns [Time, V1-V28, Amount]
        scaler: Fitted RobustScaler
    
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    df_processed = df.copy()
    df_processed[['Time', 'Amount']] = scaler.transform(df_processed[['Time', 'Amount']])
    return df_processed


def predict_fraud(model, features):
    """
    Make fraud prediction on transaction features.
    
    Parameters:
        model: Trained model
        features: DataFrame or array of shape (n_samples, 30)
    
    Returns:
        predictions: Array of binary predictions (0=legit, 1=fraud)
        probabilities: Array of fraud probabilities
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return predictions, probabilities


def predict_single_transaction(model, transaction_data, scaler):
    """
    Predict fraud for a single transaction.
    
    Parameters:
        model: Trained model
        transaction_data: Dict with keys [Time, V1-V28, Amount]
        scaler: Fitted RobustScaler
    
    Returns:
        dict: Prediction result with probability
    """
    df = pd.DataFrame([transaction_data])
    df = df[FEATURE_NAMES]
    df_processed = preprocess_data(df, scaler)
    
    prediction, probability = predict_fraud(model, df_processed)
    
    result = {
        'prediction': int(prediction[0]),
        'is_fraud': bool(prediction[0]),
        'fraud_probability': float(probability[0]),
        'confidence': 'High' if probability[0] > 0.8 or probability[0] < 0.2 else 'Medium'
    }
    
    return result


def run_demo(model, scaler):
    """Run demonstration with sample test data."""
    print("\n" + "="*60)
    print("FRAUD DETECTION PREDICTION DEMO")
    print("="*60)
    
    # Sample legitimate transaction (with realistic V values)
    legit_transaction = {
        'Time': 50000,
        'V1': -1.2, 'V2': 0.5, 'V3': -0.3, 'V4': 1.2, 'V5': -0.5,
        'V6': 0.8, 'V7': -0.2, 'V8': 0.3, 'V9': -0.1, 'V10': 0.5,
        'V11': -0.8, 'V12': 0.2, 'V13': -0.5, 'V14': 0.9, 'V15': -0.3,
        'V16': 0.4, 'V17': -0.6, 'V18': 0.1, 'V19': -0.2, 'V20': 0.3,
        'V21': -0.1, 'V22': 0.2, 'V23': -0.3, 'V24': 0.1, 'V25': -0.2,
        'V26': 0.3, 'V27': -0.1, 'V28': 0.2,
        'Amount': 50.0
    }
    
    print("\n1. Testing LEGITIMATE Transaction:")
    print("-" * 40)
    result = predict_single_transaction(model, legit_transaction, scaler)
    print(f"   Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
    print(f"   Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.2f}%)")
    print(f"   Confidence: {result['confidence']}")
    
    # Test multiple transactions
    print("\n2. Batch Prediction Test (from real data):")
    print("-" * 40)
    
    # Load some real data
    df = pd.read_csv(DATA_PATH)
    
    # Get 5 legit and 5 fraud
    legit_samples = df[df['Class'] == 0].sample(n=5, random_state=42)
    fraud_samples = df[df['Class'] == 1].sample(n=5, random_state=42)
    
    test_df = pd.concat([legit_samples, fraud_samples])
    test_features = preprocess_features(test_df[FEATURE_NAMES], scaler)
    
    predictions, probabilities = predict_fraud(model, test_features)
    
    print(f"   Total transactions: {len(predictions)}")
    print(f"   Predicted frauds: {sum(predictions)}")
    print(f"   Predicted legitimate: {len(predictions) - sum(predictions)}")
    
    print("\n   Details:")
    for i, (pred, prob, actual) in enumerate(zip(predictions, probabilities, test_df['Class'])):
        status = "FRAUD" if pred == 1 else "LEGIT"
        actual_label = "(Actual: FRAUD)" if actual == 1 else "(Actual: LEGIT)"
        print(f"   Transaction {i+1}: {status} {actual_label} (fraud_prob={prob:.4f})")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


def preprocess_features(df, scaler):
    """Preprocess features with the given scaler."""
    df_processed = df.copy()
    df_processed[['Time', 'Amount']] = scaler.transform(df_processed[['Time', 'Amount']])
    return df_processed


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Credit Card Fraud Detection - Make Predictions'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to CSV file with transactions to predict'
    )
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demo with sample data'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test with real Kaggle data'
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model()
    
    # Create and fit scaler
    scaler = create_and_fit_scaler()
    
    if args.demo or (args.input is None and not args.test):
        # Run demo
        run_demo(model, scaler)
    elif args.test:
        # Test with real data
        print("\n" + "="*60)
        print("TESTING WITH REAL DATA")
        print("="*60)
        
        df = pd.read_csv(DATA_PATH)
        
        # Sample 10 legit and 10 fraud
        legit = df[df['Class'] == 0].sample(n=10, random_state=42)
        fraud = df[df['Class'] == 1].sample(n=10, random_state=42)
        
        test_data = pd.concat([legit, fraud])
        test_features = preprocess_features(test_data[FEATURE_NAMES], scaler)
        
        predictions, probabilities = predict_fraud(model, test_features)
        
        correct_fraud = sum((predictions == 1) & (test_data['Class'] == 1))
        correct_legit = sum((predictions == 0) & (test_data['Class'] == 0))
        
        print(f"\nResults:")
        print(f"  Fraud detection: {correct_fraud}/10 correct")
        print(f"  Legit detection: {correct_legit}/10 correct")
        
    elif args.input:
        # Load and predict on custom data
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            return
        
        df = pd.read_csv(args.input)
        
        if all(col in df.columns for col in FEATURE_NAMES):
            df = df[FEATURE_NAMES]
        else:
            print(f"Error: CSV must contain these columns: {FEATURE_NAMES}")
            return
        
        df_processed = preprocess_data(df, scaler)
        predictions, probabilities = predict_fraud(model, df_processed)
        
        print(f"\nResults for {len(predictions)} transactions:")
        print(f"  Frauds detected: {sum(predictions)}")
        print(f"  Legitimate: {len(predictions) - sum(predictions)}")
        
        df['Prediction'] = predictions
        df['Fraud_Probability'] = probabilities
        
        output_file = 'outputs/predictions.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
