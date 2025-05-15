import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

# Calculating the lengths of approximation (cA) and detail (cD) coefficients
def get_wavedec_coeff_lengths(signal_length, wavelet, level, mode='symmetric'):
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
    
    if level < 0:
        raise ValueError(f"Decomposition level must be >= 0, got {level}")
    
    dummy_signal = np.zeros(signal_length)
    if level == 0:
        cA, cD = pywt.dwt(dummy_signal, wavelet, mode=mode)
        len_cA, len_cD = len(cA), len(cD)
    else:
        coeffs = pywt.wavedec(dummy_signal, wavelet, level=level, mode=mode)
        len_cA, len_cD = len(coeffs[0]), len(coeffs[1])
    print(f"get_wavedec_coeff_lengths: signal_length={signal_length}, wavelet={wavelet.name}, level={level}, mode={mode}, len_cA={len_cA}, len_cD={len_cD}")
    return len_cA, len_cD

# Function to load M4 Daily train and test data
def load_m4_daily(train_file_path, test_file_path, max_length):
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

class DownsampleTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3, **kwargs):
        super(DownsampleTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1)
        self.local_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu", kernel_regularizer=regularizers.l2(0.03)),
             layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.03))]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
        self.importance_scorer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.03))
        self.bn = layers.BatchNormalization()
        self.residual_proj = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.03))

    def call(self, inputs, training=None):
        print(f"DownsampleTransformerBlock input shape: {inputs.shape}")
        norm1 = self.layernorm1(inputs)
        attn_output = self.att(norm1, norm1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        norm2 = self.layernorm2(out1)
        local_attn_output = self.local_att(norm2, norm2)
        local_attn_output = self.dropout2(local_attn_output, training=training)
        out2 = out1 + local_attn_output
        norm3 = self.layernorm3(out2)
        ffn_output = self.ffn(norm3)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = out2 + ffn_output
        importance_scores = self.importance_scorer(out3)
        importance_scores = self.dropout3(importance_scores, training=training)
        reduced_seq_len = tf.shape(out3)[1] // 2
        reduced_seq_len = tf.maximum(reduced_seq_len, 1)
        top_k_indices = tf.math.top_k(tf.squeeze(importance_scores, axis=-1), k=reduced_seq_len)[1]
        top_k_indices = tf.sort(top_k_indices, axis=-1)
        downsampled_output = tf.gather(out3, top_k_indices, batch_dims=1)
        residual = self.residual_proj(inputs)
        residual_downsampled = tf.gather(residual, top_k_indices, batch_dims=1)
        downsampled_output = downsampled_output + residual_downsampled
        downsampled_output = self.bn(downsampled_output, training=training)
        print(f"After attention-based downsampling and BN: {downsampled_output.shape}")
        return downsampled_output

    def compute_output_shape(self, input_shape):
        seq_len = input_shape[1]
        reduced_seq_len = seq_len // 2 if seq_len is not None else None
        reduced_seq_len = max(reduced_seq_len, 1) if reduced_seq_len is not None else None
        return (input_shape[0], reduced_seq_len, self.embed_dim)

    def get_config(self):
        config = super(DownsampleTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

class TimeSeriesEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(TimeSeriesEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.value_emb = layers.Dense(embed_dim)

    def get_sinusoidal_pos_encoding(self, maxlen, embed_dim):
        position = tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / embed_dim))
        pos_encoding = position * div_term
        pos_encoding = tf.concat([tf.sin(pos_encoding), tf.cos(pos_encoding)], axis=-1)
        pos_encoding = pos_encoding[:, :embed_dim]
        return pos_encoding[tf.newaxis, :, :]

    def call(self, x):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)
        value_embeddings = self.value_emb(x)
        pos_embeddings = self.get_sinusoidal_pos_encoding(self.maxlen, self.embed_dim)
        return value_embeddings + pos_embeddings

    def get_config(self):
        config = super(TimeSeriesEmbedding, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim,
        })
        return config
    
def build_detail_transformer(input_seq_len, embed_dim, num_heads, ff_dim, num_transformer_blocks=3):
    inputs = layers.Input(shape=(input_seq_len, 1))
    x = TimeSeriesEmbedding(maxlen=input_seq_len, embed_dim=embed_dim)(inputs)
    print(f"After TimeSeriesEmbedding: {x.shape}")
    for i in range(num_transformer_blocks):
        x = DownsampleTransformerBlock(embed_dim, num_heads, ff_dim, rate=0.3)(x)
        print(f"After DownsampleTransformerBlock {i+1}: {x.shape}")
    output_seq_len = input_seq_len // (2 ** num_transformer_blocks)
    if output_seq_len < 1:
        output_seq_len = 1
    print(f"build_detail_transformer: input_seq_len={input_seq_len}, output_seq_len={output_seq_len}")
    x = layers.Conv1D(filters=1, kernel_size=1, padding='same')(x)
    print(f"After Conv1D: {x.shape}")
    x = layers.Flatten()(x)
    print(f"After Flatten: {x.shape}")
    x = layers.Dense(output_seq_len, kernel_regularizer=regularizers.l2(0.03))(x)
    print(f"After Dense: {x.shape}")
    model = keras.Model(inputs=inputs, outputs=x)
    return model

class WaveletDownsamplingModel(keras.Model):
    def __init__(self, detail_transformer_model, wavelet_name, approx_ds_factor,
                 original_length, signal_coeffs_len, dwt_level=1, normalize_details=False,
                 decomposition_mode='symmetric', **kwargs):
        super(WaveletDownsamplingModel, self).__init__(**kwargs)
        self.detail_transformer = detail_transformer_model
        self.wavelet_name = wavelet_name
        self.approx_ds_factor = approx_ds_factor
        self.original_length = original_length
        self.signal_coeffs_len = signal_coeffs_len
        self.dwt_level = dwt_level
        self.normalize_details = normalize_details
        self.decomposition_mode = decomposition_mode
        self.detail_ds_len = None

        if self.normalize_details:
            self.detail_norm_layer = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        print(f"Building WaveletDownsamplingModel with input_shape: {input_shape}")
        if not self.detail_transformer.built:
            dummy_detail_input_shape = tf.TensorShape((None, self.signal_coeffs_len, 1))
            self.detail_transformer.build(dummy_detail_input_shape)
            print("Built detail_transformer within WaveletDownsamplingModel build.")
        dummy_detail_input = tf.zeros((1, self.signal_coeffs_len, 1))
        detail_output = self.detail_transformer(dummy_detail_input)
        self.detail_ds_len = detail_output.shape[-1]
        print(f"Determined detail_transformer output length: {self.detail_ds_len}")
        if self.detail_ds_len is None:
            raise ValueError("detail_transformer output length is None.")
        len_cA, len_cD = get_wavedec_coeff_lengths(
            self.original_length, self.wavelet_name, self.dwt_level, self.decomposition_mode
        )
        print(f"Wavelet coefficients: len_cA={len_cA}, len_cD={len_cD}")
        if self.approx_ds_factor > 1:
            actual_approx_ds_len = (len_cA - self.approx_ds_factor) // self.approx_ds_factor + 1
        else:
            actual_approx_ds_len = len_cA
        print(f"actual_approx_ds_len={actual_approx_ds_len}, detail_ds_len={self.detail_ds_len}")
        self.combined_ds_len = actual_approx_ds_len + self.detail_ds_len
        print(f"Actual combined downsampled length: {self.combined_ds_len}")
        super(WaveletDownsamplingModel, self).build(input_shape)

    def call(self, original_signals_batch, training=None, **kwargs):
        print(f"Calling WaveletDownsamplingModel with input shape: {original_signals_batch.shape}")
        if len(original_signals_batch.shape) == 3 and original_signals_batch.shape[-1] == 1:
            input_for_pyfunc = tf.squeeze(original_signals_batch, axis=-1)
        else:
            input_for_pyfunc = original_signals_batch
        approx_coeffs, detail_coeffs = tf.py_function(
            func=self._decompose_batch_py_func,
            inp=[input_for_pyfunc],
            Tout=[tf.float32, tf.float32]
        )
        len_cA, _ = get_wavedec_coeff_lengths(
            self.original_length, self.wavelet_name, self.dwt_level, self.decomposition_mode
        )
        approx_coeffs.set_shape([None, len_cA])
        detail_coeffs.set_shape([None, self.signal_coeffs_len])
        if self.approx_ds_factor > 1:
            approx_coeffs_reshaped = tf.expand_dims(approx_coeffs, axis=-1)
            approx_downsampled_pooled = tf.nn.avg_pool1d(
                approx_coeffs_reshaped,
                ksize=self.approx_ds_factor,
                strides=self.approx_ds_factor,
                padding='VALID'
            )
            approx_downsampled = tf.squeeze(approx_downsampled_pooled, axis=-1)
        else:
            approx_downsampled = approx_coeffs
        print(f"approx_downsampled shape: {approx_downsampled.shape}")
        detail_coeffs_reshaped = tf.expand_dims(detail_coeffs, axis=-1)
        print(f"detail_coeffs_reshaped shape: {detail_coeffs_reshaped.shape}")
        if self.normalize_details:
            detail_coeffs_reshaped = self.detail_norm_layer(detail_coeffs_reshaped, training=training)
        detail_downsampled = self.detail_transformer(detail_coeffs_reshaped, training=training)
        print(f"detail_downsampled shape: {detail_downsampled.shape}")
        combined_downsampled = tf.concat([approx_downsampled, detail_downsampled], axis=1)
        print(f"combined_downsampled shape: {combined_downsampled.shape}")
        return combined_downsampled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.combined_ds_len)

    def _decompose_batch_py_func(self, signal_batch_tensor):
        signal_batch_np = signal_batch_tensor.numpy()
        if len(signal_batch_np.shape) == 3 and signal_batch_np.shape[-1] == 1:
            signal_batch_np = np.squeeze(signal_batch_np, axis=-1)
        approx_coeffs_list = []
        detail_coeffs_list = []
        for row in signal_batch_np:
            if len(row.shape) > 1:
                row = np.squeeze(row)
            if self.dwt_level == 0:
                cA, cD = pywt.dwt(row, self.wavelet_name, mode=self.decomposition_mode)
            else:
                coeffs = pywt.wavedec(row, self.wavelet_name, level=self.dwt_level, mode=self.decomposition_mode)
                cA = coeffs[0]
                cD = coeffs[1]
            approx_coeffs_list.append(cA)
            detail_coeffs_list.append(cD)
        return np.array(approx_coeffs_list, dtype=np.float32), np.array(detail_coeffs_list, dtype=np.float32)

    def get_config(self):
        config = super(WaveletDownsamplingModel, self).get_config()
        config.update({
            'wavelet_name': self.wavelet_name,
            'approx_ds_factor': self.approx_ds_factor,
            'original_length': self.original_length,
            'signal_coeffs_len': self.signal_coeffs_len,
            'dwt_level': self.dwt_level,
            'normalize_details': self.normalize_details,
            'decomposition_mode': self.decomposition_mode
        })
        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(
            detail_transformer_model=None,
            wavelet_name=config['wavelet_name'],
            approx_ds_factor=config['approx_ds_factor'],
            original_length=config['original_length'],
            signal_coeffs_len=config['signal_coeffs_len'],
            dwt_level=config['dwt_level'],
            normalize_details=config['normalize_details'],
            decomposition_mode=config['decomposition_mode']
        )
        return instance

def frequency_domain_loss(y_true, y_pred):
    y_true_fft = tf.abs(tf.signal.fft(tf.cast(y_true, tf.complex64)))
    y_pred_fft = tf.abs(tf.signal.fft(tf.cast(y_pred, tf.complex64)))
    return tf.reduce_mean(tf.square(y_true_fft - y_pred_fft))

def downsampling_loss(y_true, y_pred):
    mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
    freq_loss = frequency_domain_loss(y_true, y_pred)
    return mse_loss + 0.5 * freq_loss

def main():
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    base_path = "."
    train_file = os.path.join(base_path, "M4/Daily/Daily-train.csv")
    test_file = os.path.join(base_path, "M4/Daily/Daily-test.csv")

    embed_dim = 64
    num_heads = 12
    ff_dim = 64
    num_transformer_blocks = 3
    wavelet_name = 'db4'
    dwt_level = 1
    approx_ds_factor = 2
    normalize_details = True
    decomposition_mode = 'symmetric'
    batch_size = 128
    epochs = 20
    learning_rate = 0.0001

    # Compute dynamic original_length from training data
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found at: {train_file}")
    train_df = pd.read_csv(train_file)
    lengths = [len(row.iloc[1:].dropna()) for _, row in train_df.iterrows()]
    original_length = int(np.median(lengths))
    print(f"Computed dynamic original_length: {original_length}")

    # Compute wavelet coefficient lengths
    len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')
    print(f"Computed signal_coeffs_len: {signal_coeffs_len}")

    # Load M4 Daily data
    X_train, X_test = load_m4_daily(train_file, test_file, max_length=original_length)
    
    # Normalize data
    data_mean = np.mean(X_train, axis=0)
    data_std = np.std(X_train, axis=0)
    data_std = np.where(data_std == 0, 1, data_std)  # Avoid division by zero
    X_train_normalized = (X_train - data_mean) / data_std
    X_test_normalized = (X_test - data_mean) / data_std

    # Build and initialize model
    detail_transformer = build_detail_transformer(signal_coeffs_len, embed_dim, num_heads, ff_dim, num_transformer_blocks)
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

    # Generate targets (self-supervised: downsampled output as target)
    y_train = model.call(X_train_normalized, training=False)
    y_test = model.call(X_test_normalized, training=False)

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss=downsampling_loss)
    model.build(input_shape=(None, original_length))
    print("Wavelet Downsampling Model Summary:")
    model.summary(line_length=150)

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)

    # Train model
    print("--- Training Wavelet Downsampling Model for 250 epochs ---")
    history = model.fit(
        X_train_normalized, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_normalized, y_test),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Evaluate model
    print("\n--- Evaluating Model ---")
    mse_train = mean_squared_error(y_train.numpy(), model.predict(X_train_normalized, verbose=0))
    mse_test = mean_squared_error(y_test.numpy(), model.predict(X_test_normalized, verbose=0))
    print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")

    # Downsampled example
    print("\n--- Downsampled Representation Example ---")
    sample_input = X_test_normalized[:10]
    downsampled_representation = model.call(sample_input, training=False)
    print("Downsampled Representation Shape:", downsampled_representation.shape)

    # Save models
    print("\n--- Saving Models ---")
    model.save('downsampling_model.keras')
    detail_transformer.save('detail_transformer_model.keras')

    # Plot results
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    downsampled_signal = model.call(X_test_normalized[:1], training=False)[0].numpy()
    num_downsampled_samples = len(downsampled_signal)
    plt.plot(downsampled_signal, label='Downsampled Signal', alpha=0.7, marker='o')
    plt.title(f'Downsampled Time Series\n({num_downsampled_samples} Samples)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(1, 3, 3)
    original_signal = X_test_normalized[0]
    num_original_samples = len(original_signal)
    plt.plot(original_signal, label='Original Signal', alpha=0.7, color='orange')
    plt.title(f'Original M4 Daily Signal\n({num_original_samples} Samples)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()