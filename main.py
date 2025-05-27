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
import psutil
import logging
from core.downsampling_algorithm import WaveletDownsamplingModel, TimeSeriesEmbedding, DownsampleTransformerBlock, get_wavedec_coeff_lengths, downsampling_loss, build_detail_transformer
from core.streaming_pipeline import setup_streaming_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log', mode='w', delay=True)
    ]
)
logger = logging.getLogger(__name__)

def load_m4_daily(train_file_path, test_file_path, max_length=150):
    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        raise FileNotFoundError(f"M4 Daily dataset files not found at: {train_file_path} or {test_file_path}")

    # Load training data
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

    logger.info(f"Loaded M4 Daily data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    return X_train, X_test

def augment_data(X, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, X.shape)
    X_augmented = X + noise
    return X_augmented

def mixup_data(X, y, alpha=0.2):
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D inputs, got X with {X.ndim} dims and y with {y.ndim} dims")
    batch_size = X.shape[0]
    indices = np.random.permutation(batch_size)
    lam = np.random.beta(alpha, alpha, batch_size)
    X_mixed = lam[:, None] * X + (1 - lam[:, None]) * X[indices]
    y_mixed = lam[:, None] * y + (1 - lam[:, None]) * y[indices]
    return X_mixed, y_mixed

def non_stream_pipeline(args):
    # Configuration
    embed_dim = 64
    num_heads = 8
    ff_dim = 64
    num_transformer_blocks = 4
    wavelet_name = 'db4'
    dwt_level = 1
    retention_rate = 0.8
    approx_ds_factor = 2
    original_length = 150
    len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')
    normalize_details = True
    decomposition_mode = 'symmetric'
    batch_size = 32
    epochs = 1
    learning_rate = 0.0001

    # Monitor memory usage
    logger.info(f"Memory usage before model build: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    # Test TimeSeriesEmbedding
    logger.info("Testing TimeSeriesEmbedding with dummy input...")
    dummy_input = tf.zeros((25, signal_coeffs_len, 1))
    embedding_layer = TimeSeriesEmbedding(maxlen=signal_coeffs_len, embed_dim=embed_dim)
    dummy_output = embedding_layer(dummy_input)
    logger.info(f"Dummy TimeSeriesEmbedding output shape: {dummy_output.shape}")

    # Load and normalize data
    X_train, X_test = load_m4_daily(args.train_file, args.test_file, max_length=original_length)
    data_mean = np.nanmean(X_train, axis=0)
    data_std = np.nanstd(X_train, axis=0)
    data_std = np.where(data_std == 0, 1, data_std)
    X_train_normalized = np.where(np.isnan(X_train), 0, (X_train - data_mean) / data_std)
    X_test_normalized = np.where(np.isnan(X_test), 0, (X_test - data_mean) / data_std)

    # Build model
    detail_transformer = build_detail_transformer(
        signal_coeffs_len, embed_dim, num_heads, ff_dim, num_transformer_blocks, retention_rate=retention_rate
    )
    model = WaveletDownsamplingModel(
        detail_transformer_model=detail_transformer,
        wavelet_name=wavelet_name,
        approx_ds_factor=approx_ds_factor,
        original_length=original_length,
        signal_coeffs_len=signal_coeffs_len,
        dwt_level=dwt_level,
        normalize_details=normalize_details,
        decomposition_mode=decomposition_mode
    )
    model.build(input_shape=(None, original_length))

    logger.info(f"Memory usage after model build: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    # Data augmentation
    X_train_augmented = augment_data(X_train_normalized)
    X_train_mixed, _ = mixup_data(X_train_normalized, X_train_normalized)
    X_train_combined = np.concatenate([X_train_normalized, X_train_augmented, X_train_mixed], axis=0)
    logger.info(f"X_train_combined shape: {X_train_combined.shape}")

    # Generate targets
    y_train_combined = model.call(X_train_combined, training=False, return_indices=False)
    y_test_normalized = model.call(X_test_normalized, training=False, return_indices=False)
    logger.info(f"y_train_combined shape: {y_train_combined.shape}")
    logger.info(f"y_test_normalized shape: {y_test_normalized.shape}")

    # Compile and train model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss=downsampling_loss)
    logger.info("Wavelet Downsampling Model Summary:")
    model.summary(line_length=150)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)

    logger.info("Training Wavelet Downsampling Model for 5 epochs")
    history = model.fit(
        X_train_combined, y_train_combined,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_normalized, y_test_normalized),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Evaluate model
    logger.info("Evaluating Model")
    mse_train = mean_squared_error(y_train_combined.numpy(), model.predict(X_train_combined, verbose=0))
    mse_test = mean_squared_error(y_test_normalized.numpy(), model.predict(X_test_normalized, verbose=0))
    logger.info(f"Train MSE: {mse_train}, Test MSE: {mse_test}")

    # Downsampled example
    logger.info("Downsampled Representation Example")
    sample_input = X_test_normalized[:10]
    downsampled_representation, indices_list = model.call(sample_input, training=False, return_indices=True)
    logger.info(f"Downsampled Representation Shape: {downsampled_representation.shape}")

    # Save models
    logger.info("Saving Models")
    model.save('downsampling_model.keras')
    detail_transformer.save('detail_transformer_model.keras')

    # Statistical evaluation
    logger.info("Statistical Evaluation of Downsampled Representation")
    original_signal = X_test_normalized[1]
    downsampled_signal = model.call(X_test_normalized[:1], training=False, return_indices=False)[0].numpy()
    _, indices_list = model.call(X_test_normalized[:1], training=False, return_indices=True)
    approx_indices = indices_list[0][0].numpy()
    detail_indices = indices_list[-1][0].numpy()
    len_cA, len_cD = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')
    detail_indices_mapped = (detail_indices * (original_length / len_cD)).astype(int)
    selected_indices = np.concatenate([approx_indices, detail_indices_mapped])
    selected_indices = np.clip(selected_indices, 0, len(original_signal) - 1)
    selected_indices = np.unique(selected_indices)
    selected_values = original_signal[selected_indices]
    logger.info(f"Number of unique downsampled points: {len(selected_indices)}")
    logger.info(f"Selected indices: {selected_indices}")
    logger.info(f"Selected values shape: {selected_values.shape}")

    # Reconstruct signal
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
    logger.info(f"Reconstructed signal shape: {reconstructed_signal.shape}")

    # Compute metrics
    mse = mean_squared_error(original_signal, reconstructed_signal)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_signal, reconstructed_signal)
    r2 = r2_score(original_signal, reconstructed_signal)
    correlation, _ = pearsonr(original_signal, reconstructed_signal)
    original_fft = np.abs(np.fft.fft(original_signal))
    reconstructed_fft = np.abs(np.fft.fft(reconstructed_signal))
    spectral_mse = mean_squared_error(original_fft, reconstructed_fft)

    logger.info(f"MSE between original and reconstructed: {mse:.6f}")
    logger.info(f"RMSE between original and reconstructed: {rmse:.6f}")
    logger.info(f"MAE between original and reconstructed: {mae:.6f}")
    logger.info(f"R-squared score: {r2:.6f}")
    logger.info(f"Pearson correlation coefficient: {correlation:.6f}")
    logger.info(f"Spectral MSE (frequency domain): {spectral_mse:.6f}")

    # Plot results
    plt.figure(figsize=(18, 10))
    plt.subplot(3, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 2, 2)
    num_downsampled_samples = len(downsampled_signal)
    plt.plot(range(num_downsampled_samples), downsampled_signal, label='Downsampled Signal', alpha=0.7, marker='o')
    plt.title(f'Downsampled Time Series\n({num_downsampled_samples} Samples)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 2, 3)
    num_original_samples = len(original_signal)
    plt.plot(original_signal, label='Original Signal', alpha=0.7, color='orange')
    plt.title(f'Original M4 Daily Signal\n({num_original_samples} Samples)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(original_signal, label='Original Signal', alpha=0.7, color='orange')
    plt.scatter(selected_indices, selected_values, color='blue', label='Downsampled Points', s=50, alpha=0.8)
    plt.title('Original Signal with Downsampled Points')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(original_signal, label='Original Signal', alpha=0.7, color='orange')
    plt.plot(reconstructed_signal, label='Reconstructed Signal', alpha=0.7, color='green', linestyle='--')
    plt.title('Original vs Reconstructed Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.savefig('downsampling_results.png')
    plt.close()

def stream_pipeline(args):
    # Configuration
    embed_dim = 8  # Reduced from 16
    num_heads = 1  # Reduced from 2
    ff_dim = 8     # Reduced from 16
    num_transformer_blocks = 2
    wavelet_name = 'db4'
    dwt_level = 1
    retention_rate = 0.8
    approx_ds_factor = 2
    original_length = 150
    len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')
    normalize_details = True
    decomposition_mode = 'symmetric'

    # Monitor memory usage
    logger.info(f"Memory usage before model build: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    # Load data for streaming
    X_train, X_test = load_m4_daily(args.train_file, args.test_file, max_length=original_length)
    data_mean = np.nanmean(X_train, axis=0)
    data_std = np.nanstd(X_train, axis=0)
    data_std = np.where(data_std == 0, 1, data_std)
    X_train_normalized = np.where(np.isnan(X_train), 0, (X_train - data_mean) / data_std)
    X_test_normalized = np.where(np.isnan(X_test), 0, (X_test - data_mean) / data_std)

    # Build model
    logger.info("Building detail_transformer")
    detail_transformer = build_detail_transformer(
        signal_coeffs_len, embed_dim, num_heads, ff_dim, num_transformer_blocks, retention_rate=retention_rate
    )
    logger.info("Building WaveletDownsamplingModel")
    model = WaveletDownsamplingModel(
        detail_transformer_model=detail_transformer,
        wavelet_name=wavelet_name,
        approx_ds_factor=approx_ds_factor,
        original_length=original_length,
        signal_coeffs_len=signal_coeffs_len,
        dwt_level=dwt_level,
        normalize_details=normalize_details,
        decomposition_mode=decomposition_mode
    )
    model.build(input_shape=(None, original_length))

    logger.info(f"Memory usage after model build: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    # Test model
    logger.info("Testing model with sample input")
    sample_input = X_test_normalized[:1].reshape(1, original_length, 1)
    try:
        sample_output = model.call(sample_input, training=False, return_indices=False)
        logger.info(f"Sample output shape: {sample_output.shape}, first 10 values: {sample_output.numpy().flatten()[:10]}")
        if tf.reduce_any(tf.math.is_nan(sample_output)) or tf.reduce_any(tf.math.is_inf(sample_output)):
            logger.error("NaN or Inf in sample output")
            raise ValueError("Model produces invalid output")
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        raise

    # Run streaming pipeline
    logger.info("Running Streaming Pipeline")
    setup_streaming_pipeline(X_train_normalized, X_test_normalized, model, original_length, input_topic='m4-input-topic', output_topic='m4-downsampled-topic')

def main():
    parser = argparse.ArgumentParser(description="Run the downsampling pipeline in stream or non-stream mode.")
    parser.add_argument('--pipeline', choices=['stream', 'non-stream'], default='non-stream', help="Choose pipeline mode: 'stream' or 'non-stream'")
    parser.add_argument('--train_file', default="M4/Daily/Daily-train.csv", help="Path to training CSV file")
    parser.add_argument('--test_file', default="M4/Daily/Daily-test.csv", help="Path to test CSV file")
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    np.random.seed(42)

    # Limit TensorFlow threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    if args.pipeline == 'stream':
        stream_pipeline(args)
    else:
        non_stream_pipeline(args)

if __name__ == "__main__":
    main()