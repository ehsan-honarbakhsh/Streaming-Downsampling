
import os
import numpy as np
import pandas as pd
def load_m4_daily(train_file_path, test_file_path, max_length=150):
    """
    Load M4 Daily dataset from Daily-train.csv and Daily-test.csv.
    
    Parameters:
    - train_file_path: Path to Daily-train.csv
    - test_file_path: Path to Daily-test.csv
    - max_length: Target length for padding/truncating series
    
    Returns:
    - X_train: Numpy array of shape (n_train_samples, max_length)
    - X_test: Numpy array of shape (n_test_samples, max_length)
    """
    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        raise FileNotFoundError(f"M4 Daily dataset files not found at: {train_file_path} or {test_file_path}")

    # Load training data
    train_df = pd.read_csv(train_file_path)
    X_train = []
    for _, row in train_df.iterrows():
        # Extract values (skip 'id' column)
        series = row.iloc[1:].dropna().values.astype(float)
        # Truncate or pad to max_length
        if len(series) > max_length:
            series = series[:max_length]
        else:
            series = np.pad(series, (0, max_length - len(series)), mode='constant', constant_values=0)
        X_train.append(series)
    X_train = np.array(X_train)

    # Load test data
    test_df = pd.read_csv(test_file_path)
    X_test = []
    for _, row in test_df.iterrows():
        series = row.iloc[1:].dropna().values.astype(float)
        if len(series) > max_length:
            series = series[:max_length]
        else:
            series = np.pad(series, (0, max_length - len(series)), mode='constant', constant_values=0)
        X_test.append(series)
    X_test = np.array(X_test)

    print(f"Loaded M4 Daily data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    return X_train, X_test

xx=load_m4_daily('M4/Daily/Daily-train.csv', 'M4/Daily/Daily-test.csv')