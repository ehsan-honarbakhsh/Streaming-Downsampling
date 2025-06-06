import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import os
import argparse
import logging
from core.downsampling_algorithm import WaveletDownsamplingModel, TimeSeriesEmbedding, DownsampleTransformerBlock, get_wavedec_coeff_lengths, downsampling_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluate.log', mode='w', delay=True)
    ]
)
logger = logging.getLogger(__name__)

def load_m4_daily(train_file_path, test_file_path, max_length=150):
    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        raise FileNotFoundError(f"M4 Daily dataset files not found at: {train_file_path} or {test_file_path}")

    # Loading training data
    train_df = pd.read_csv(train_file_path)
    X_train = []
    for _, row in train_df.iterrows():
        series = row.iloc[1:].dropna().values.astype(float)
        if len(series) > max_length:
            series = series[:max_length]
        else:
            series = np.pad(series, (0, max_length - len(series)), mode='constant', constant_values=0)
        X_train.append(series)
    X_train = np.array(X_train)

    # Loading test data
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

    logger.info(f"Loaded M4 Daily data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    return X_train, X_test

def evaluate_model(args):
    original_length = 200
    wavelet_name = 'db4'
    dwt_level = 1

    # Load data
    X_train, X_test = load_m4_daily(args.train_file, args.test_file, max_length=original_length)
    data_mean = np.nanmean(X_train, axis=0)
    data_std = np.nanstd(X_train, axis=0)
    data_std = np.where(data_std == 0, 1, data_std)
    X_train_normalized = np.where(np.isnan(X_train), 0, (X_train - data_mean) / data_std)
    X_test_normalized = np.where(np.isnan(X_test), 0, (X_test - data_mean) / data_std)

    # Load the trained model
    model_path = 'downsampling_model.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    logger.info("Loading trained model...")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "WaveletDownsamplingModel": WaveletDownsamplingModel,
            "TimeSeriesEmbedding": TimeSeriesEmbedding,
            "DownsampleTransformerBlock": DownsampleTransformerBlock,
            "downsampling_loss": downsampling_loss
        }
    )
    logger.info("Model loaded successfully")

    # Generate targets for evaluation
    y_train_combined = model.call(X_train_normalized, training=False, return_indices=False)
    y_test_normalized = model.call(X_test_normalized, training=False, return_indices=False)
    logger.info(f"y_train_combined shape: {y_train_combined.shape}")
    logger.info(f"y_test_normalized shape: {y_test_normalized.shape}")

    # Evaluate the model on downsampled outputs
    logger.info("Evaluating model on downsampled outputs")
    mse_train = mean_squared_error(y_train_combined.numpy(), model.predict(X_train_normalized, verbose=0))
    mse_test = mean_squared_error(y_test_normalized.numpy(), model.predict(X_test_normalized, verbose=0))
    logger.info(f"Train MSE: {mse_train:.6f}")
    logger.info(f"Test MSE: {mse_test:.6f}")

    # Statistical evaluation on reconstructed signals
    logger.info("Performing statistical evaluation on reconstructed signals for entire test dataset")
    mse_list, rmse_list, mae_list, r2_list, corr_list, spectral_mse_list = [], [], [], [], [], []
    len_cA, len_cD = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')

    # Process test data in batches to manage memory
    batch_size = 32
    num_samples = X_test_normalized.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = X_test_normalized[start_idx:end_idx]
        batch_downsampled, batch_indices_list = model.call(batch, training=False, return_indices=True)
        batch_downsampled = batch_downsampled.numpy()

        for i in range(batch.shape[0]):
            original_signal = batch[i]
            downsampled_signal = batch_downsampled[i]
            approx_indices = batch_indices_list[0][i].numpy()
            detail_indices = batch_indices_list[-1][i].numpy()
            detail_indices_mapped = (detail_indices * (original_length / len_cD)).astype(int)
            selected_indices = np.concatenate([approx_indices, detail_indices_mapped])
            selected_indices = np.clip(selected_indices, 0, len(original_signal) - 1)
            selected_indices = np.unique(selected_indices)
            selected_values = original_signal[selected_indices]

            # Reconstruct the signal
            x_full = np.arange(original_length)
            sorted_idx = np.argsort(selected_indices)
            sorted_indices = selected_indices[sorted_idx]
            sorted_values = selected_values[sorted_idx]
            if sorted_indices[0] != 0:
                sorted_indices = np.insert(sorted_indices, 0, 0)
                sorted_values = np.insert(sorted_values, 0, original_signal[0])
            if sorted_indices[-1] != original_length - 1:
                sorted_indices = np.append(sorted_indices, original_length - 1)
                sorted_values = np.append(sorted_values, original_signal[-1])
            interpolator = interp1d(sorted_indices, sorted_values, kind='linear', fill_value="extrapolate")
            reconstructed_signal = interpolator(x_full)

            # Compute metrics
            mse = mean_squared_error(original_signal, reconstructed_signal)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(original_signal, reconstructed_signal)
            r2 = r2_score(original_signal, reconstructed_signal)
            correlation, _ = pearsonr(original_signal, reconstructed_signal)
            original_fft = np.abs(np.fft.fft(original_signal))
            reconstructed_fft = np.abs(np.fft.fft(reconstructed_signal))
            spectral_mse = mean_squared_error(original_fft, reconstructed_fft)

            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)
            corr_list.append(correlation)
            spectral_mse_list.append(spectral_mse)

            logger.info(f"Sample {start_idx + i + 1}/{num_samples}: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}, Corr={correlation:.6f}, Spectral MSE={spectral_mse:.6f}")

    # Compute average and standard deviation of metrics
    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    avg_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    avg_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)
    avg_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)
    avg_corr = np.mean(corr_list)
    std_corr = np.std(corr_list)
    avg_spectral_mse = np.mean(spectral_mse_list)
    std_spectral_mse = np.std(spectral_mse_list)

    logger.info("Average Evaluation Metrics Across Test Dataset:")
    logger.info(f"  MSE: Mean={avg_mse:.6f}, Std={std_mse:.6f}")
    logger.info(f"  RMSE: Mean={avg_rmse:.6f}, Std={std_rmse:.6f}")
    logger.info(f"  MAE: Mean={avg_mae:.6f}, Std={std_mae:.6f}")
    logger.info(f"  R-squared: Mean={avg_r2:.6f}, Std={std_r2:.6f}")
    logger.info(f"  Pearson Correlation: Mean={avg_corr:.6f}, Std={std_corr:.6f}")
    logger.info(f"  Spectral MSE: Mean={avg_spectral_mse:.6f}, Std={std_spectral_mse:.6f}")

    # Visualize a subset of results (first 3 samples to avoid clutter)
    logger.info("Generating visualization for a subset of samples")
    num_visualize = min(3, num_samples)
    plt.figure(figsize=(18, 10))
    for i in range(num_visualize):
        original_signal = X_test_normalized[i]
        downsampled_signal = model.call(X_test_normalized[i:i+1], training=False, return_indices=False)[0].numpy()
        _, indices_list = model.call(X_test_normalized[i:i+1], training=False, return_indices=True)
        approx_indices = indices_list[0][0].numpy()
        detail_indices = indices_list[-1][0].numpy()
        detail_indices_mapped = (detail_indices * (original_length / len_cD)).astype(int)
        selected_indices = np.concatenate([approx_indices, detail_indices_mapped])
        selected_indices = np.clip(selected_indices, 0, len(original_signal) - 1)
        selected_indices = np.unique(selected_indices)
        selected_values = original_signal[selected_indices]

        # Reconstruct the signal
        x_full = np.arange(original_length)
        sorted_idx = np.argsort(selected_indices)
        sorted_indices = selected_indices[sorted_idx]
        sorted_values = selected_values[sorted_idx]
        if sorted_indices[0] != 0:
            sorted_indices = np.insert(sorted_indices, 0, 0)
            sorted_values = np.insert(sorted_values, 0, original_signal[0])
        if sorted_indices[-1] != original_length - 1:
            sorted_indices = np.append(sorted_indices, original_length - 1)
            sorted_values = np.append(sorted_values, original_signal[-1])
        interpolator = interp1d(sorted_indices, sorted_values, kind='linear', fill_value="extrapolate")
        reconstructed_signal = interpolator(x_full)

        # Plotting
        plt.subplot(num_visualize, 3, i * 3 + 1)
        plt.plot(range(len(downsampled_signal)), downsampled_signal, label='Downsampled Signal', alpha=0.7, marker='o')
        plt.title(f'Downsampled Time Series (Sample {i+1})\n({len(downsampled_signal)} Samples)')
        plt.xlabel('Sample')
        plt.ylabel('Standardised Value')
        plt.legend()

        plt.subplot(num_visualize, 3, i * 3 + 2)
        plt.plot(original_signal, label='Original Signal', alpha=0.7, color='orange')
        plt.scatter(selected_indices, selected_values, color='blue', label='Downsampled Points', s=50, alpha=0.8)
        plt.title(f'Original Signal with Downsampled Points (Sample {i+1})')
        plt.xlabel('Sample')
        plt.ylabel('Standardised Value')
        plt.legend()

        plt.subplot(num_visualize, 3, i * 3 + 3)
        plt.plot(original_signal, label='Original Signal', alpha=0.7, color='orange')
        plt.plot(reconstructed_signal, label='Reconstructed Signal', alpha=0.7, color='green', linestyle='--')
        plt.title(f'Original vs Reconstructed Signal (Sample {i+1})')
        plt.xlabel('Sample')
        plt.ylabel('Standardised Value')
        plt.legend()

    plt.tight_layout()
    plt.savefig('downsampling_results.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate the downsampling model on the entire test dataset.")
    parser.add_argument('--train_file', default="M4/Daily/Daily-train.csv", help="Path to training CSV file")
    parser.add_argument('--test_file', default="M4/Daily/Daily-test.csv", help="Path to test CSV file")
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    np.random.seed(42)

    # Limiting TensorFlow threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    evaluate_model(args)

if __name__ == "__main__":
    main()

# How to run the script:
#python Only_downsamplingmodel_evaluation.py --train_file M4/Daily/Daily-train.csv --test_file M4/Daily/Daily-test.csv  