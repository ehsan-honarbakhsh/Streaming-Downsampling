import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import os
import argparse
import psutil
import logging
from core.downsampling_algorithm2 import WaveletDownsamplingModel, TimeSeriesEmbedding, DownsampleTransformerBlock, get_wavedec_coeff_lengths, downsampling_loss, build_detail_transformer
from core.streaming_pipeline import setup_streaming_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log', mode='w', delay=True)
    ]
)
logger = logging.getLogger(__name__)

def load_m4_datasets(train_files, test_files, max_length=200):
    """Load M4 datasets (Daily, Hourly, Minute) for training and testing.

    Args:
        train_files (list): List of paths to training CSV files.
        test_files (list): List of paths to test CSV files.
        max_length (int): Maximum sequence length for padding/truncating.

    Returns:
        tuple: (X_train, X_test) as numpy arrays.
    """
    X_train_all = []
    X_test_all = []

    for train_file, test_file in zip(train_files, test_files):
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Dataset files not found: {train_file} or {test_file}")

        # Load training data
        train_df = pd.read_csv(train_file)
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
        test_df = pd.read_csv(test_file)
        X_test = []
        for _, row in test_df.iterrows():
            series = row.iloc[1:].dropna().values.astype(float)
            if len(series) > max_length:
                series = series[:max_length]
            else:
                series = np.pad(series, (0, max_length - len(series)), mode='constant', constant_values=0)
            X_test.append(series)
        X_test = np.array(X_test)

        # Normalize each dataset separately
        data_mean = np.nanmean(X_train, axis=0)
        data_std = np.nanstd(X_train, axis=0)
        data_std = np.where(data_std == 0, 1, data_std)
        X_train_normalized = np.where(np.isnan(X_train), 0, (X_train - data_mean) / data_std)
        X_test_normalized = np.where(np.isnan(X_test), 0, (X_test - data_mean) / data_std)

        X_train_all.append(X_train_normalized)
        X_test_all.append(X_test_normalized)

        logger.info(f"Loaded {os.path.basename(train_file)}: X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    # Concatenate all datasets
    X_train_combined = np.concatenate(X_train_all, axis=0)
    X_test_combined = np.concatenate(X_test_all, axis=0)

    logger.info(f"Combined M4 data: X_train_combined shape={X_train_combined.shape}, X_test_combined shape={X_test_combined.shape}")
    return X_train_combined, X_test_combined

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
    embed_dim = 64
    num_heads = 4
    ff_dim = 64
    num_transformer_blocks = 1
    wavelet_name = 'db4'
    dwt_level = 1
    retention_rate = 0.8
    approx_ds_factor = 2
    original_length = 200
    len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')
    normalize_details = True
    decomposition_mode = 'symmetric'
    batch_size = 32
    epochs = 2
    learning_rate = 0.0001

    # Monitoring memory usage
    logger.info(f"Memory usage before model build: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    logger.info("Testing TimeSeriesEmbedding with dummy input...")
    dummy_input = tf.zeros((25, signal_coeffs_len, 1))
    embedding_layer = TimeSeriesEmbedding(maxlen=signal_coeffs_len, embed_dim=embed_dim)
    dummy_output = embedding_layer(dummy_input)
    logger.info(f"Dummy TimeSeriesEmbedding output shape: {dummy_output.shape}")

    # Load all M4 datasets
    #, args.monthly_train,args.quarterly_train,args.weekly_train,args.yearly_train,args.daily_train,
    train_files = [args.hourly_train]
    test_files = [args.hourly_test]
    #,args.monthly_test,args.quarterly_test,args.weekly_test,args.yearly_test,args.daily_test,
    X_train, X_test = load_m4_datasets(train_files, test_files, max_length=original_length)

    # Building the model
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
    X_train_augmented = augment_data(X_train)
    X_train_mixed, _ = mixup_data(X_train, X_train)
    X_train_combined = np.concatenate([X_train, X_train_augmented, X_train_mixed], axis=0)
    logger.info(f"X_train_combined shape: {X_train_combined.shape}")

    # Generate targets
    y_train_combined = model.call(X_train_combined, training=False, return_indices=False)
    y_test = model.call(X_test, training=False, return_indices=False)
    logger.info(f"y_train_combined shape: {y_train_combined.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    # Compile and train the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss=downsampling_loss)
    logger.info("Wavelet Downsampling Model Summary:")
    model.summary(line_length=150)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)

    logger.info("Training Wavelet Downsampling Model for 2 epochs")
    history = model.fit(
        X_train_combined, y_train_combined,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Saving the models
    logger.info("Saving Models")
    model.save('downsampling_model.keras')
    detail_transformer.save('detail_transformer_model.keras')

def stream_pipeline(args):
    embed_dim = 64
    num_heads = 8
    ff_dim = 64
    num_transformer_blocks = 1
    wavelet_name = 'db4'
    dwt_level = 1
    retention_rate = 0.8
    approx_ds_factor = 2
    original_length = 200
    len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')
    normalize_details = True
    decomposition_mode = 'symmetric'

    # Monitoring memory usage
    logger.info(f"Memory usage before model build: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    # Load all M4 datasets
    #args.daily_train,
    train_files = [args.hourly_train]
    test_files = [args.hourly_test]
    #args.daily_test,
    X_train, X_test = load_m4_datasets(train_files, test_files, max_length=original_length)

    # Building the model
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

    logger.info("Testing model with sample input")
    sample_input = X_test[:1].reshape(1, original_length, 1)
    try:
        sample_output = model.call(sample_input, training=False, return_indices=False)
        logger.info(f"Sample output shape: {sample_output.shape}, first 10 values: {sample_output.numpy().flatten()[:10]}")
        if tf.reduce_any(tf.math.is_nan(sample_output)) or tf.reduce_any(tf.math.is_inf(sample_output)):
            logger.error("NaN or Inf in sample output")
            raise ValueError("Model produces invalid output")
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        raise

    # Running streaming pipeline
    logger.info("Running Streaming Pipeline")
    setup_streaming_pipeline(X_train, X_test, model, original_length, input_topic='m4-input-topic', output_topic='m4-downsampled-topic')

def main():
    parser = argparse.ArgumentParser(description="Run the downsampling pipeline in stream or non-stream mode.")
    parser.add_argument('--pipeline', choices=['stream', 'non-stream'], default='non-stream', help="Choose pipeline mode: 'stream' or 'non-stream'")
    #parser.add_argument('--daily_train', default="M4/Daily/Daily-train.csv", help="Path to Daily training CSV file")
    #parser.add_argument('--daily_test', default="M4/Daily/Daily-test.csv", help="Path to Daily test CSV file")
    parser.add_argument('--hourly_train', default="M4/Hourly/Hourly-train.csv", help="Path to Hourly training CSV file")
    parser.add_argument('--hourly_test', default="M4/Hourly/Hourly-test.csv", help="Path to Hourly test CSV file")
    #parser.add_argument('--monthly_train', default="M4/Monthly/Monthly-train.csv", help="Path to Monthly training CSV file")
    #parser.add_argument('--monthly_test', default="M4/Monthly/Monthly-test.csv", help="Path to Monthly test CSV file")
    #parser.add_argument('--quarterly_train', default="M4/Quarterly/Quarterly-train.csv", help="Path to Quarterly training CSV file")
    #parser.add_argument('--quarterly_test', default="M4/Quarterly/Quarterly-test.csv", help="Path to Quarterly test CSV file")
    #parser.add_argument('--weekly_train', default="M4/Weekly/Weekly-train.csv", help="Path to Weekly training CSV file")
    #parser.add_argument('--weekly_test', default="M4/Weekly/Weekly-test.csv", help="Path to Weekly test CSV file")
    #parser.add_argument('--yearly_train', default="M4/Yearly/Yearly-train.csv", help="Path to Yearly training CSV file")
    #parser.add_argument('--yearly_test', default="M4/Yearly/Yearly-test.csv", help="Path to Yearly test CSV file")
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    np.random.seed(42)

    # Limiting TensorFlow threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    if args.pipeline == 'stream':
        stream_pipeline(args)
    else:
        non_stream_pipeline(args)

if __name__ == "__main__":
    main()