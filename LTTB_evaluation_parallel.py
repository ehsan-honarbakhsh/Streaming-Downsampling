import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import os
import argparse
import logging
import time
try:
    from tsdownsample import LTTBDownsampler
except ImportError:
    raise ImportError("tsdownsample is not installed. Please install it using: pip install tsdownsample")
from core.downsampling_algorithm import (
    WaveletDownsamplingModel,
    TimeSeriesEmbedding,
    DownsampleTransformerBlock,
    get_wavedec_coeff_lengths,
    downsampling_loss
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_2.log', mode='w', delay=True)
    ]
)
logger = logging.getLogger(__name__)

def load_m4_daily_data(train_file_path, test_file_path, max_length=150):
    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        raise FileNotFoundError(f"M4 Daily dataset files not found at: {train_file_path} or {test_file_path}")

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

def uniform_downsampling(signal, target_length):
    original_length = len(signal)
    step = original_length // target_length
    indices = np.arange(0, original_length, step)[:target_length]
    indices = np.clip(indices, 0, original_length - 1)
    values = signal[indices]
    return indices, values

def average_pooling(signal, target_length):
    original_length = len(signal)
    window_size = max(1, original_length // target_length)
    indices = np.arange(0, original_length, window_size)[:target_length]
    indices = np.clip(indices, 0, original_length - 1)
    values = []
    for i in indices:
        window = signal[i:i + window_size]
        values.append(np.mean(window) if len(window) > 0 else signal[i])
    return indices, np.array(values)

def max_pooling(signal, target_length):
    original_length = len(signal)
    window_size = max(1, original_length // target_length)
    indices = np.arange(0, original_length, window_size)[:target_length]
    indices = np.clip(indices, 0, original_length - 1)
    values = []
    for i in indices:
        window = signal[i:i + window_size]
        values.append(np.max(window) if len(window) > 0 else signal[i])
    return indices, np.array(values)

def random_sampling(signal, target_length):
    original_length = len(signal)
    indices = np.random.choice(original_length, size=target_length, replace=False)
    indices = np.sort(indices)
    values = signal[indices]
    return indices, values

def lttb_downsampling(signal, target_length):
    n = len(signal)
    if target_length >= n:
        return np.arange(n), signal
    if target_length < 2:
        return np.array([0]), np.array([signal[0]])

    downsampler = LTTBDownsampler()
    indices = downsampler.downsample(signal, n_out=target_length)
    indices = indices.astype(int)
    indices = np.clip(indices, 0, n - 1)
    values = signal[indices]
    return indices, values

def reconstruct_signal(original_signal, indices, values, original_length):
    x_full = np.arange(original_length)
    sorted_idx = np.argsort(indices)
    sorted_indices = indices[sorted_idx]
    sorted_values = values[sorted_idx]
    if sorted_indices[0] != 0:
        sorted_indices = np.insert(sorted_indices, 0, 0)
        sorted_values = np.insert(sorted_values, 0, original_signal[0])
    if sorted_indices[-1] != original_length - 1:
        sorted_indices = np.append(sorted_indices, original_length - 1)
        sorted_values = np.append(sorted_values, original_signal[-1])
    interpolator = interp1d(sorted_indices, sorted_values, kind='linear', fill_value="extrapolate")
    reconstructed_signal = interpolator(x_full)
    return reconstructed_signal

def compute_metrics(original_signal, reconstructed_signal):
    mse = mean_squared_error(original_signal, reconstructed_signal)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_signal, reconstructed_signal)
    r2 = r2_score(original_signal, reconstructed_signal)
    correlation, _ = pearsonr(original_signal, reconstructed_signal)
    original_fft = np.abs(np.fft.fft(original_signal))
    reconstructed_fft = np.abs(np.fft.fft(reconstructed_signal))
    spectral_mse = mean_squared_error(original_fft, reconstructed_fft)
    return mse, rmse, mae, r2, correlation, spectral_mse

def process_sample(i, original_signal, batch_downsampled, batch_indices_list, original_length, len_cD, target_length, sample_idx_global):
    metrics = {
        'WaveletDownsampling': {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0, 'corr': 0, 'spectral_mse': 0, 'execution_time': 0},
        'Uniform': {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0, 'corr': 0, 'spectral_mse': 0, 'execution_time': 0},
        'AveragePooling': {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0, 'corr': 0, 'spectral_mse': 0, 'execution_time': 0},
        'MaxPooling': {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0, 'corr': 0, 'spectral_mse': 0, 'execution_time': 0},
        'RandomSampling': {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0, 'corr': 0, 'spectral_mse': 0, 'execution_time': 0},
        'LTTB': {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0, 'corr': 0, 'spectral_mse': 0, 'execution_time': 0}
    }

    # WaveletDownsampling
    start_time = time.perf_counter()
    downsampled_signal = batch_downsampled[i]
    approx_indices = batch_indices_list[0][i].numpy()
    detail_indices = batch_indices_list[-1][i].numpy()
    detail_indices_mapped = (detail_indices * (original_length / len_cD)).astype(int)
    selected_indices = np.concatenate([approx_indices, detail_indices_mapped])
    selected_indices = np.clip(selected_indices, 0, original_length - 1)
    selected_indices = np.unique(selected_indices)
    selected_values = original_signal[selected_indices]
    reconstructed = reconstruct_signal(original_signal, selected_indices, selected_values, original_length)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    mse, rmse, mae, r2, corr, spectral_mse = compute_metrics(original_signal, reconstructed)
    metrics['WaveletDownsampling'] = {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr, 'spectral_mse': spectral_mse, 'execution_time': execution_time
    }
    logger.info(f"Sample {sample_idx_global + 1}: WaveletDownsampling: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}, Corr={corr:.6f}, Spectral MSE={spectral_mse:.6f}, Execution Time={execution_time:.6f}s")

    # Other downsampling methods
    for method, func in [
        ('Uniform', uniform_downsampling),
        ('AveragePooling', average_pooling),
        ('MaxPooling', max_pooling),
        ('RandomSampling', random_sampling),
        ('LTTB', lttb_downsampling)
    ]:
        start_time = time.perf_counter()
        indices, values = func(original_signal, target_length)
        reconstructed = reconstruct_signal(original_signal, indices, values, original_length)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        mse, rmse, mae, r2, corr, spectral_mse = compute_metrics(original_signal, reconstructed)
        metrics[method] = {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr, 'spectral_mse': spectral_mse, 'execution_time': execution_time
        }
        logger.info(f"Sample {sample_idx_global + 1}: {method}: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}, Corr={corr:.6f}, Spectral MSE={spectral_mse:.6f}, Execution Time={execution_time:.6f}s")

    return metrics

def evaluate_model(args):
    original_length = 200
    wavelet_name = 'db4'
    dwt_level = 1
    target_length = 43

    X_train, X_test = load_m4_daily_data(args.train_file, args.test_file, max_length=original_length)
    data_mean = np.nanmean(X_train, axis=0)
    data_std = np.nanstd(X_train, axis=0)
    data_std = np.where(data_std == 0, 1, data_std)
    X_train_normalized = np.where(np.isnan(X_train), 0, (X_train - data_mean) / data_std)
    X_test_normalized = np.where(np.isnan(X_test), 0, (X_test - data_mean) / data_std)

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

    methods = ['WaveletDownsampling', 'Uniform', 'AveragePooling', 'MaxPooling', 'RandomSampling', 'LTTB']
    metrics = {method: {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'corr': [], 'spectral_mse': [], 'execution_time': []} for method in methods}
    len_cA, len_cD = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')

    batch_size = 32
    num_samples = X_test_normalized.shape[0]
    logger.info(f"Evaluating {num_samples} test samples across {len(methods)} methods")
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = X_test_normalized[start_idx:end_idx]
        batch_downsampled, batch_indices_list = model.call(batch, training=False, return_indices=True)
        batch_downsampled = batch_downsampled.numpy()

        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(process_sample)(
                i, batch[i], batch_downsampled, batch_indices_list, original_length, len_cD, target_length, start_idx + i
            ) for i in range(batch.shape[0])
        )

        for sample_metrics in results:
            for method in methods:
                for metric_name, value in sample_metrics[method].items():
                    metrics[method][metric_name].append(value)

    logger.info("Average Evaluation Metrics Across Test Dataset:")
    for method in methods:
        logger.info(f"\nMethod: {method}")
        for metric_name in ['mse', 'rmse', 'mae', 'r2', 'corr', 'spectral_mse', 'execution_time']:
            mean_val = np.mean(metrics[method][metric_name])
            std_val = np.std(metrics[method][metric_name])
            unit = 's' if metric_name == 'execution_time' else ''
            logger.info(f"  {metric_name.upper()}: Mean={mean_val:.6f}{unit}, Std={std_val:.6f}{unit}")

    logger.info("Generating visualization for a subset of samples")
    num_visualize = min(3, num_samples)
    plt.figure(figsize=(18, 10 * num_visualize))
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
        wavelet_reconstructed = reconstruct_signal(original_signal, selected_indices, selected_values, original_length)

        uniform_indices, uniform_values = uniform_downsampling(original_signal, target_length)
        avg_pool_indices, avg_pool_values = average_pooling(original_signal, target_length)
        max_pool_indices, max_pool_values = max_pooling(original_signal, target_length)
        random_indices, random_values = random_sampling(original_signal, target_length)
        lttb_indices, lttb_values = lttb_downsampling(original_signal, target_length)
        uniform_reconstructed = reconstruct_signal(original_signal, uniform_indices, uniform_values, original_length)
        avg_pool_reconstructed = reconstruct_signal(original_signal, avg_pool_indices, avg_pool_values, original_length)
        max_pool_reconstructed = reconstruct_signal(original_signal, max_pool_indices, max_pool_values, original_length)
        random_reconstructed = reconstruct_signal(original_signal, random_indices, random_values, original_length)
        lttb_reconstructed = reconstruct_signal(original_signal, lttb_indices, lttb_values, original_length)

        plt.subplot(num_visualize, 2, i * 2 + 1)
        plt.plot(original_signal, label='Original Signal', alpha=0.7, color='orange')
        plt.scatter(selected_indices, selected_values, color='blue', label='Wavelet Points', s=50, alpha=0.8)
      #  plt.scatter(uniform_indices, uniform_values, color='green', label='Uniform Points', s=50, alpha=0.8, marker='x')
       # plt.scatter(avg_pool_indices, avg_pool_values, color='red', label='Avg Pool Points', s=50, alpha=0.8, marker='^')
       # plt.scatter(max_pool_indices, max_pool_values, color='purple', label='Max Pool Points', s=50, alpha=0.8, marker='s')
        #plt.scatter(random_indices, random_values, color='cyan', label='Random Points', s=50, alpha=0.8, marker='o')
        plt.scatter(lttb_indices, lttb_values, color='magenta', label='LTTB Points', s=50, alpha=0.8, marker='*')
        plt.title(f'Original Signal with Downsampled Points (Sample {i+1})')
        plt.xlabel('Sample')
        plt.ylabel('Standardized Value')
        plt.legend()

        plt.subplot(num_visualize, 2, i * 2 + 2)
        plt.plot(original_signal, label='Original Signal', alpha=0.7, color='orange')
        plt.plot(wavelet_reconstructed, label='Wavelet Reconstructed', alpha=0.7, color='blue', linestyle='--')
       # plt.plot(uniform_reconstructed, label='Uniform Reconstructed', alpha=0.7, color='green', linestyle='-.')
       # plt.plot(avg_pool_reconstructed, label='Avg Pool Reconstructed', alpha=0.7, color='red', linestyle=':')
       # plt.plot(max_pool_reconstructed, label='Max Pool Reconstructed', alpha=0.7, color='purple', linestyle='-')
       #plt.plot(random_reconstructed, label='Random Reconstructed', alpha=0.7, color='cyan', linestyle='--')
        plt.plot(lttb_reconstructed, label='LTTB Reconstructed', alpha=0.7, color='magenta', linestyle='-.')
        plt.title(f'Original vs Reconstructed Signals (Sample {i+1})')
        plt.xlabel('Sample')
        plt.ylabel('Standardized Value')
        plt.legend()

    plt.tight_layout()
    plt.savefig('downsampling_comparison.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare downsampling model with other methods on the entire test dataset.")
    parser.add_argument('--train_file', default="M4/Daily/Daily-train.csv", help="Path to training CSV file")
    parser.add_argument('--test_file', default="M4/Daily/Daily-test.csv", help="Path to test CSV file")
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    np.random.seed(42)

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    evaluate_model(args)

if __name__ == "__main__":
    main()

#python LTTB_evaluation_parallel.py --train_file M4/Daily/Daily-train.csv --test_file M4/Daily/Daily-test.csv  