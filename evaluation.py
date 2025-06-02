import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import logging
from core.downsampling_algorithm import WaveletDownsamplingModel,TimeSeriesEmbedding,DownsampleTransformerBlock,downsampling_loss ,build_detail_transformer, get_wavedec_coeff_lengths
from main import load_m4_daily



# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid conflicts
logger.handlers = []

# Create and add FileHandler
file_handler = logging.FileHandler('benchmark.log', mode='w', delay=False)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Create and add StreamHandler for console output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

def uniform_sampling(signal, target_length):
    """Downsample a signal by uniformly selecting points."""
    original_length = len(signal)
    indices = np.linspace(0, original_length - 1, target_length, dtype=int)
    downsampled = signal[indices]
    return downsampled, indices

def average_pooling(signal, target_length):
    """Downsample a signal using average pooling."""
    original_length = len(signal)
    window_size = max(1, original_length // target_length)
    downsampled = []
    indices = []
    for i in range(0, original_length, window_size):
        window = signal[i:i + window_size]
        if len(window) > 0:
            downsampled.append(np.mean(window))
            indices.append(i + window_size // 2)  # Middle of the window
    downsampled = np.array(downsampled)[:target_length]
    indices = np.array(indices)[:target_length]
    if len(downsampled) < target_length:
        downsampled = np.pad(downsampled, (0, target_length - len(downsampled)), mode='constant')
        indices = np.pad(indices, (0, target_length - len(indices)), mode='constant')
    return downsampled, indices

def reconstruct_signal(original_signal, downsampled_signal, indices, original_length):
    """Reconstruct the signal using linear interpolation."""
    sorted_idx = np.argsort(indices)
    sorted_indices = indices[sorted_idx]
    sorted_values = downsampled_signal[sorted_idx]
    if sorted_indices[0] != 0:
        sorted_indices = np.insert(sorted_indices, 0, 0)
        sorted_values = np.insert(sorted_values, 0, original_signal[0])
    if sorted_indices[-1] != original_length - 1:
        sorted_indices = np.append(sorted_indices, original_length - 1)
        sorted_values = np.append(sorted_values, original_signal[-1])
    interpolator = interp1d(sorted_indices, sorted_values, kind='linear', fill_value="extrapolate")
    return interpolator(np.arange(original_length))

def evaluate_method(signals, method, method_name, target_length, original_length, len_cD=None):
    """Evaluate a downsampling method on a set of signals."""
    logger.info(f"Starting evaluation for {method_name}")
    mse_list, rmse_list, inference_times = [], [], []
    
    for i, signal in enumerate(signals):
        start_time = time.time()
        if method_name == "Wavelet-Transformer":
            if len_cD is None:
                raise ValueError("len_cD must be provided for Wavelet-Transformer method")
            signal_input = signal.reshape(1, original_length, 1)
            with tf.device('/CPU:0'):
                downsampled, indices_list = method(tf.convert_to_tensor(signal_input, dtype=tf.float32), training=False, return_indices=True)
            downsampled = downsampled.numpy().flatten()
            # Map detail indices to original signal domain
            approx_indices = indices_list[0][0].numpy()
            detail_indices = indices_list[-1][0].numpy()
            detail_indices_mapped = (detail_indices * (original_length / len_cD)).astype(int)
            indices = np.concatenate([approx_indices, detail_indices_mapped])
            indices = np.clip(np.unique(indices), 0, original_length - 1)
        else:
            downsampled, indices = method(signal, target_length)
        inference_time = time.time() - start_time
        
        reconstructed = reconstruct_signal(signal, downsampled, indices, original_length)
        mse = mean_squared_error(signal, reconstructed)
        rmse = np.sqrt(mse)
        
        mse_list.append(mse)
        rmse_list.append(rmse)
        inference_times.append(inference_time)
        
        # Log progress for every 10 signals
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(signals)} signals for {method_name}")
    
    result = {
        'method': method_name,
        'mse_mean': np.mean(mse_list),
        'mse_std': np.std(mse_list),
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'inference_time_mean': np.mean(inference_times) * 1000,  # Convert to ms
        'inference_time_std': np.std(inference_times) * 1000
    }
    logger.info(f"Completed evaluation for {method_name}: MSE={result['mse_mean']:.6f}±{result['mse_std']:.6f}, "
                f"RMSE={result['rmse_mean']:.6f}±{result['rmse_std']:.6f}, "
                f"Inference Time={result['inference_time_mean']:.2f}±{result['inference_time_std']:.2f}ms")
    for handler in logger.handlers:
        handler.flush()  # Ensure logs are written
    return result

def plot_comparison(results, metric, title, ylabel, filename):
    """Plot comparison of methods for a given metric."""
    methods = [r['method'] for r in results]
    means = [r[f'{metric}_mean'] for r in results]
    stds = [r[f'{metric}_std'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(title)
    plt.xlabel('Method')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved plot: {filename}")
    for handler in logger.handlers:
        handler.flush()

def main():
    # Hyperparameters
    original_length = 150
    wavelet_name = 'db4'
    dwt_level = 1
    retention_rate = 0.8
    approx_ds_factor = 2
    embed_dim = 64
    num_heads = 8
    ff_dim = 64
    num_transformer_blocks = 4
    normalize_details = True
    decomposition_mode = 'symmetric'

    # Verify log file creation
    logger.info("Starting benchmark script")
    try:
        with open('benchmark.log', 'w') as f:
            f.write("Benchmark script started\n")
    except Exception as e:
        logger.error(f"Failed to create benchmark.log: {e}")
        raise

    # Load data
    train_file = "M4/Daily/Daily-train.csv"
    test_file = "M4/Daily/Daily-test.csv"
    logger.info(f"Loading data from {train_file} and {test_file}")
    X_train, X_test = load_m4_daily(train_file, test_file, max_length=original_length)
    
    # Normalize data
    data_mean = np.nanmean(X_train, axis=0)
    data_std = np.nanstd(X_train, axis=0)
    data_std = np.where(data_std == 0, 1, data_std)
    X_test_normalized = np.where(np.isnan(X_test), 0, (X_test - data_mean) / data_std)
    logger.info(f"Normalized test data shape: {X_test_normalized.shape}")
    
    # Compute wavelet coefficient lengths
    len_cA, len_cD = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, decomposition_mode)
    actual_approx_ds_len = (len_cA - approx_ds_factor) // approx_ds_factor + 1 if approx_ds_factor > 1 else len_cA
    detail_ds_len = int(len_cD * (retention_rate ** num_transformer_blocks))
    target_length = actual_approx_ds_len + detail_ds_len
    logger.info(f"Target downsampled length: {target_length}, len_cA: {len_cA}, len_cD: {len_cD}")

    # Build wavelet-transformer model
    logger.info("Building detail_transformer")
    detail_transformer = build_detail_transformer(
        len_cD, embed_dim, num_heads, ff_dim, num_transformer_blocks, retention_rate
    )
    logger.info("Building WaveletDownsamplingModel")
    model = WaveletDownsamplingModel(
        detail_transformer_model=detail_transformer,
        wavelet_name=wavelet_name,
        approx_ds_factor=approx_ds_factor,
        original_length=original_length,
        signal_coeffs_len=len_cD,
        dwt_level=dwt_level,
        normalize_details=normalize_details,
        decomposition_mode=decomposition_mode
    )
    model.build(input_shape=(None, original_length, 1))
    logger.info("Model built successfully")

    # Load pre-trained model weights if available
    model_path = 'downsampling_model.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'WaveletDownsamplingModel': WaveletDownsamplingModel,
                'TimeSeriesEmbedding': TimeSeriesEmbedding,
                'DownsampleTransformerBlock': DownsampleTransformerBlock,
                'downsampling_loss': downsampling_loss
            }
        )
        logger.info(f"Loaded pre-trained model from {model_path}")
    else:
        logger.warning("No pre-trained model found. Using untrained model.")

    # Define methods to compare
    methods = [
        (model, "Wavelet-Transformer", len_cD),
        (lambda x, tl=target_length: uniform_sampling(x, tl), "Uniform Sampling", None),
        (lambda x, tl=target_length: average_pooling(x, tl), "Average Pooling", None)
    ]

    # Evaluate methods
    results = []
    try:
        for method, method_name, len_cD_param in methods:
            logger.info(f"Evaluating {method_name}")
            result = evaluate_method(X_test_normalized[:100], method, method_name, target_length, original_length, len_cD=len_cD_param)
            results.append(result)
            logger.info(f"{method_name} results: MSE={result['mse_mean']:.6f}±{result['mse_std']:.6f}, "
                        f"RMSE={result['rmse_mean']:.6f}±{result['rmse_std']:.6f}, "
                        f"Inference Time={result['inference_time_mean']:.2f}±{result['inference_time_std']:.2f}ms")
            for handler in logger.handlers:
                handler.flush()
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        for handler in logger.handlers:
            handler.flush()
        raise

    # Plot results
    plot_comparison(results, 'mse', 'Mean Squared Error Comparison', 'MSE', 'mse_comparison.png')
    plot_comparison(results, 'rmse', 'Root Mean Squared Error Comparison', 'RMSE', 'rmse_comparison.png')
    plot_comparison(results, 'inference_time', 'Inference Time Comparison', 'Time (ms)', 'inference_time_comparison.png')
    logger.info("Benchmarking completed")
    for handler in logger.handlers:
        handler.flush()

if __name__ == "__main__":
    main()