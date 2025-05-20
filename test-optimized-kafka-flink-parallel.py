import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import os
import json
from kafka import KafkaProducer
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink, KafkaRecordSerializationSchema
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.common import WatermarkStrategy, Configuration
from pyflink.common.serialization import DeserializationSchema, SerializationSchema
from pyflink.java_gateway import get_gateway
from threading import Lock
from pyflink.datastream.connectors.kafka import DeliveryGuarantee
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

# Updated function to load M4 Daily train and test data
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

    print(f"Loaded M4 Daily data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    return X_train, X_test

@keras.saving.register_keras_serializable()
class DownsampleTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3, **kwargs):
        super(DownsampleTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
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

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape (batch_size, seq_len, embed_dim), got {input_shape}")
        
        attention_input_shape = input_shape
        self.att.build(query_shape=attention_input_shape, 
                      value_shape=attention_input_shape, 
                      key_shape=attention_input_shape)
        self.local_att.build(query_shape=attention_input_shape, 
                            value_shape=attention_input_shape, 
                            key_shape=attention_input_shape)
        
        self.ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.layernorm3.build(input_shape)
        self.importance_scorer.build(input_shape)
        self.bn.build(input_shape)
        self.residual_proj.build(input_shape)
        self.built = True

    def call(self, inputs, training=None):
        print(f"DownsampleTransformerBlock input shape: {inputs.shape}")
        tf.ensure_shape(inputs, [None, None, self.embed_dim])
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
        print(f"Importance scores shape: {importance_scores.shape}")
        tf.ensure_shape(importance_scores, [None, None, 1])
        importance_scores_squeezed = tf.squeeze(importance_scores, axis=-1)
        print(f"Importance scores squeezed shape: {importance_scores_squeezed.shape}")
        tf.ensure_shape(importance_scores_squeezed, [None, None])
        reduced_seq_len = tf.shape(out3)[1] // 2
        reduced_seq_len = tf.maximum(reduced_seq_len, 1)
        top_k_indices = tf.math.top_k(importance_scores_squeezed, k=reduced_seq_len)[1]
        print(f"Top k indices shape: {top_k_indices.shape}")
        tf.ensure_shape(top_k_indices, [None, None])
        top_k_indices = tf.sort(top_k_indices, axis=-1)
        downsampled_output = tf.gather(out3, top_k_indices, batch_dims=1, axis=1)
        print(f"Downsampled output shape: {downsampled_output.shape}")
        tf.ensure_shape(downsampled_output, [None, None, self.embed_dim])
        residual = self.residual_proj(inputs)
        residual_downsampled = tf.gather(residual, top_k_indices, batch_dims=1, axis=1)
        print(f"Residual downsampled shape: {residual_downsampled.shape}")
        tf.ensure_shape(residual_downsampled, [None, None, self.embed_dim])
        downsampled_output = downsampled_output + residual_downsampled
        downsampled_output = self.bn(downsampled_output, training=training)
        print(f"After attention-based downsampling and BN: {downsampled_output.shape}")
        tf.ensure_shape(downsampled_output, [None, None, self.embed_dim])
        return downsampled_output, top_k_indices

    def compute_output_shape(self, input_shape):
        seq_len = input_shape[1]
        reduced_seq_len = seq_len // 2 if seq_len is not None else None
        reduced_seq_len = max(reduced_seq_len, 1) if reduced_seq_len is not None else None
        return [(input_shape[0], reduced_seq_len, self.embed_dim), (input_shape[0], reduced_seq_len)]

    def get_config(self):
        config = super(DownsampleTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

@keras.saving.register_keras_serializable()
class TimeSeriesEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(TimeSeriesEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1, embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(embed_dim,),
            initializer='zeros',
            trainable=True
        )

    def build(self, input_shape):
        if len(input_shape) != 3 or input_shape[1] != self.maxlen or input_shape[2] != 1:
            raise ValueError(
                f"Expected input shape (batch_size, {self.maxlen}, 1), got {input_shape}"
            )
        print(f"TimeSeriesEmbedding build input shape: {input_shape}")
        super(TimeSeriesEmbedding, self).build(input_shape)

    def get_sinusoidal_pos_encoding(self, maxlen, embed_dim):
        position = tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / embed_dim))
        pos_encoding = position * div_term
        pos_encoding = tf.concat([tf.sin(pos_encoding), tf.cos(pos_encoding)], axis=-1)
        pos_encoding = pos_encoding[:, :embed_dim]
        return pos_encoding[tf.newaxis, :, :]

    def call(self, x):
        print(f"TimeSeriesEmbedding input shape: {x.shape}")
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)
        x = tf.ensure_shape(x, [None, self.maxlen, 1])
        value_embeddings = x @ tf.expand_dims(self.kernel, axis=0)
        value_embeddings = value_embeddings + self.bias
        value_embeddings = tf.ensure_shape(value_embeddings, [None, self.maxlen, self.embed_dim])
        print(f"Value embeddings shape: {value_embeddings.shape}")
        output = value_embeddings
        output = tf.ensure_shape(output, [None, self.maxlen, self.embed_dim])
        print(f"TimeSeriesEmbedding output shape: {output.shape}")
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.maxlen, self.embed_dim)

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
    if len(x.shape) != 3 or x.shape[1] != input_seq_len or x.shape[2] != embed_dim:
        raise ValueError(
            f"Expected TimeSeriesEmbedding output shape (batch_size, {input_seq_len}, {embed_dim}), "
            f"got {x.shape}"
        )
    all_indices = []
    for i in range(num_transformer_blocks):
        print(f"Before DownsampleTransformerBlock {i+1}: {x.shape}")
        layer_output = DownsampleTransformerBlock(embed_dim, num_heads, ff_dim, rate=0.3)(x)
        x = layer_output[0]
        all_indices.append(layer_output[1])
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
    model = keras.Model(inputs=inputs, outputs=[x] + all_indices)
    return model

@keras.saving.register_keras_serializable()
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
        detail_outputs = self.detail_transformer(dummy_detail_input)
        detail_output = detail_outputs[0]
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

    def call(self, original_signals_batch, training=None, return_indices=False, **kwargs):
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
            approx_indices = tf.range(0, len_cA, self.approx_ds_factor, dtype=tf.int32)
            approx_indices = tf.expand_dims(approx_indices, axis=0)
            approx_indices = tf.tile(approx_indices, [tf.shape(approx_coeffs)[0], 1])
        else:
            approx_downsampled = approx_coeffs
            approx_indices = tf.range(0, len_cA, dtype=tf.int32)
            approx_indices = tf.expand_dims(approx_indices, axis=0)
            approx_indices = tf.tile(approx_indices, [tf.shape(approx_coeffs)[0], 1])
        print(f"approx_downsampled shape: {approx_downsampled.shape}")
        detail_coeffs_reshaped = tf.expand_dims(detail_coeffs, axis=-1)
        print(f"detail_coeffs_reshaped shape: {detail_coeffs_reshaped.shape}")
        if self.normalize_details:
            detail_norm_layer = getattr(self, 'detail_norm_layer', None)
            if detail_norm_layer is None:
                raise ValueError("detail_norm_layer is None but normalize_details is True")
            detail_coeffs_reshaped = detail_norm_layer(detail_coeffs_reshaped, training=training)
        print(f"detail_transformer type: {type(self.detail_transformer)}")
        print(f"detail_transformer callable: {callable(self.detail_transformer)}")
        if self.detail_transformer is None:
            raise ValueError("detail_transformer is None")
        if not callable(self.detail_transformer):
            raise ValueError(f"detail_transformer is not callable: {type(self.detail_transformer)}")
        detail_outputs = self.detail_transformer(detail_coeffs_reshaped, training=training)
        detail_downsampled = detail_outputs[0]
        detail_indices_list = detail_outputs[1:]
        print(f"detail_downsampled shape: {detail_downsampled.shape}")
        combined_downsampled = tf.concat([approx_downsampled, detail_downsampled], axis=1)
        print(f"combined_downsampled shape: {combined_downsampled.shape}")
        if return_indices:
            return combined_downsampled, [approx_indices] + detail_indices_list
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
            'detail_transformer': keras.saving.serialize_keras_object(self.detail_transformer),
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
        detail_transformer_config = config.pop('detail_transformer')
        detail_transformer = keras.saving.deserialize_keras_object(detail_transformer_config)
        instance = cls(
            detail_transformer_model=detail_transformer,
            wavelet_name=config['wavelet_name'],
            approx_ds_factor=config['approx_ds_factor'],
            original_length=config['original_length'],
            signal_coeffs_len=config['signal_coeffs_len'],
            dwt_level=config['dwt_level'],
            normalize_details=config['normalize_details'],
            decomposition_mode=config['decomposition_mode']
        )
        return instance

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

def frequency_domain_loss(y_true, y_pred):
    y_true_fft = tf.abs(tf.signal.fft(tf.cast(y_true, tf.complex64)))
    y_pred_fft = tf.abs(tf.signal.fft(tf.cast(y_pred, tf.complex64)))
    return tf.reduce_mean(tf.square(y_true_fft - y_pred_fft))

def downsampling_loss(y_true, y_pred):
    mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
    freq_loss = frequency_domain_loss(y_true, y_pred)
    return mse_loss + 0.5 * freq_loss

class DeserializationSchema(DeserializationSchema):
    def __init__(self):
        super(DeserializationSchema, self).__init__()
        gateway = get_gateway()
        self._j_deserialization_schema = gateway.jvm.org.apache.flink.api.common.serialization.SimpleStringSchema()

    def deserialize(self, message: bytes) -> list:
        try:
            return json.loads(message.decode('utf-8'))
        except Exception as e:
            print(f"Error deserializing message: {message}. Error: {e}")
            return []

    def open(self, context):
        pass

    def get_produced_type(self):
        return Types.LIST(Types.FLOAT())

class SerializationSchema(SerializationSchema):
    def __init__(self, topic='m4-downsampled-topic'):
        super(SerializationSchema, self).__init__()
        self.topic = topic
        gateway = get_gateway()
        j_string_serialization_schema = gateway.jvm.org.apache.flink.api.common.serialization.SimpleStringSchema()
        self._j_serialization_schema = gateway.jvm.org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema.builder() \
            .setTopic(self.topic) \
            .setValueSerializationSchema(j_string_serialization_schema) \
            .build()

    def serialize(self, element: list) -> bytes:
        try:
            json_string = json.dumps(element)
            return json_string.encode('utf-8')
        except Exception as e:
            print(f"Error serializing element: {element}. Error: {e}")
            return b''

    def open(self, context):
        pass

    def get_produced_type(self):
        return Types.LIST(Types.FLOAT())

def main():
    keras.utils.set_random_seed(42)
    np.random.seed(42)

    base_path = "."
    train_file = os.path.join(base_path, "M4/Daily/Daily-train.csv")
    test_file = os.path.join(base_path, "M4/Daily/Daily-test.csv")

    embed_dim = 64
    num_heads = 8
    ff_dim = 64
    num_transformer_blocks = 4
    wavelet_name = 'db4'
    dwt_level = 1
    approx_ds_factor = 2
    original_length = 150
    len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')
    print(f"Computed signal_coeffs_len: {signal_coeffs_len}")
    normalize_details = True
    decomposition_mode = 'symmetric'
    batch_size = 128
    epochs = 5
    learning_rate = 0.0001

    # Test TimeSeriesEmbedding with dummy input
    print("\nTesting TimeSeriesEmbedding with dummy input...")
    dummy_input = tf.zeros((25, signal_coeffs_len, 1))
    embedding_layer = TimeSeriesEmbedding(maxlen=signal_coeffs_len, embed_dim=embed_dim)
    dummy_output = embedding_layer(dummy_input)
    print(f"Dummy TimeSeriesEmbedding output shape: {dummy_output.shape}")

    # Load M4 Daily data
    X_train, X_test = load_m4_daily(train_file, test_file, max_length=original_length)
    
    # Normalize data
    data_mean = np.mean(X_train, axis=0)
    data_std = np.std(X_train, axis=0)
    data_std = np.where(data_std == 0, 1, data_std)
    X_train_normalized = (X_train - data_mean) / data_std
    X_test_normalized = (X_test - data_mean) / data_std

    # Kafka and Flink Streaming
    print("\n--- Streaming M4 Dataset with Kafka and Flink ---")
    
    # Kafka Producer: Stream M4 dataset to Kafka topic
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: str(k).encode('utf-8'),
        batch_size=16384,
        linger_ms=5,
        compression_type='gzip'
    )
    input_topic = 'm4-input-topic'
    
    # Stream training and test data
    total_series = len(X_train_normalized) + len(X_test_normalized)
    print(f"Streaming {total_series} series to Kafka topic: {input_topic}")
    for idx, series in enumerate(np.concatenate([X_train_normalized, X_test_normalized], axis=0)):
        producer.send(input_topic, key=str(idx % 4), value=series.tolist())
        if idx % 100 == 0:
            print(f"Sent {idx} series to Kafka")
    producer.flush()
    producer.close()
    print("Finished streaming M4 dataset to Kafka")

    # Build the model for streaming
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
    model.build(input_shape=(None, original_length))

    # Save the model to a file for loading in process_element
    model_path = os.path.join(base_path, "downsampling_model.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Flink Streaming Pipeline
    print("Setting up Flink streaming pipeline...")
    
    # Configure Flink to include Kafka connector and client JARs
    config = Configuration()
    config.set_string(
        "pipeline.jars",
        "file:///Users/ehsanhonarbakhsh/Documents/GitHub/Downsampling/flink-connector-kafka-4.0.0-2.0.jar;"
        "file:///Users/ehsanhonarbakhsh/Documents/GitHub/Downsampling/kafka-clients-4.0.0.jar"
    )
    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_parallelism(4)  # Match partitions and CPU cores
    
    # Kafka Source
    deserializer = DeserializationSchema()
    kafka_source = KafkaSource.builder() \
        .set_bootstrap_servers('localhost:9092') \
        .set_topics(input_topic) \
        .set_group_id('m4-flink-group') \
        .set_property("auto.offset.reset", "earliest") \
        .set_value_only_deserializer(deserializer) \
        .build()
    
    # Data Stream
    data_stream = env.from_source(kafka_source, WatermarkStrategy.no_watermarks(), "Kafka Source")
    
    # Process Stream: Per-message processing
    def _get_model():
        # Process-local model instance
        if not hasattr(_get_model, 'model'):
            print("Loading model in _get_model")
            _get_model.model = keras.models.load_model(
                os.path.join(".", "downsampling_model.keras"),
                custom_objects={
                    "WaveletDownsamplingModel": WaveletDownsamplingModel,
                    "TimeSeriesEmbedding": TimeSeriesEmbedding,
                    "DownsampleTransformerBlock": DownsampleTransformerBlock
                }
            )
            print("Model loaded successfully")
        return _get_model.model

    def process_element(element):
        # Process-local counter
        if not hasattr(process_element, 'count'):
            process_element.count = 0
        process_element.count += 1
        if process_element.count % 100 == 1:
            print(f"Processing element {process_element.count}")
        
        try:
            if isinstance(element, str):
                print("Received string input, attempting to parse as JSON")
                element = json.loads(element)
            if not isinstance(element, list):
                raise ValueError(f"Expected list, got {type(element)}")
            if not all(isinstance(x, (int, float)) for x in element):
                raise ValueError("Element contains non-numeric values")
            
            series = np.array(element, dtype=np.float32)
            if series.shape[0] != original_length:
                if series.shape[0] > original_length:
                    series = series[:original_length]
                else:
                    series = np.pad(series, (0, original_length - series.shape[0]), mode='constant', constant_values=0)
            model_input = series.reshape(1, original_length, 1)
            print(f"process_element: model_input shape={model_input.shape}")
            
            model = _get_model()
            if model is None:
                print("Error: model is None")
                return json.dumps([])
            if not callable(getattr(model, 'call', None)):
                print(f"Error: model.call is not callable: {type(model)}")
                return json.dumps([])
                
            downsampled = model.call(tf.convert_to_tensor(model_input), training=False, return_indices=False)
            print(f"process_element: downsampled shape={downsampled.shape}")
            
            return json.dumps(downsampled.numpy().flatten().tolist())
        except Exception as e:
            print(f"Error processing element: {e}")
            return json.dumps([])

    processed_stream = data_stream.map(process_element, output_type=Types.STRING())
    
    # Kafka Sink
    output_topic = 'm4-downsampled-topic'
    serializer = SerializationSchema(topic=output_topic)
    kafka_sink = KafkaSink.builder() \
        .set_bootstrap_servers('localhost:9092') \
        .set_record_serializer(serializer) \
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE) \
        .build()
    
    processed_stream.sink_to(kafka_sink)
    
    print(f"Executing Flink pipeline to process {total_series} elements...")
    env.execute("M4 Downsampling Pipeline")
    print(f"Processed approximately {total_series} elements")
        
    # Data augmentation
    X_train_augmented = augment_data(X_train_normalized)
    X_train_mixed, _ = mixup_data(X_train_normalized, X_train_normalized)
    X_train_combined = np.concatenate([X_train_normalized, X_train_augmented, X_train_mixed], axis=0)
    print(f"X_train_combined shape: {X_train_combined.shape}")

    # Generate targets (self-supervised)
    y_train_combined = model.call(X_train_combined, training=False, return_indices=False)
    y_test_normalized = model.call(X_test_normalized, training=False, return_indices=False)
    print(f"y_train_combined shape: {y_train_combined.shape}")
    print(f"y_test_normalized shape: {y_test_normalized.shape}")

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
    print("--- Training Wavelet Downsampling Model for 5 epochs ---")
    history = model.fit(
        X_train_combined, y_train_combined,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_normalized, y_test_normalized),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Evaluate model
    print("\n--- Evaluating Model ---")
    mse_train = mean_squared_error(y_train_combined.numpy(), model.predict(X_train_combined, verbose=0))
    mse_test = mean_squared_error(y_test_normalized.numpy(), model.predict(X_test_normalized, verbose=0))
    print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")

    # Downsampled example
    print("\n--- Downsampled Representation Example ---")
    sample_input = X_test_normalized[:10]
    downsampled_representation, indices_list = model.call(sample_input, training=False, return_indices=True)
    print("Downsampled Representation Shape:", downsampled_representation.shape)

    # Save models
    print("\n--- Saving Models ---")
    model.save('downsampling_model.keras')
    detail_transformer.save('detail_transformer_model.keras')

    # Statistical evaluation
    print("\n--- Statistical Evaluation of Downsampled Representation ---")
    original_signal = X_test_normalized[0]
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
    print(f"Number of unique downsampled points: {len(selected_indices)}")
    print(f"Selected indices: {selected_indices}")
    print(f"Selected values shape: {selected_values.shape}")

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
    print(f"Reconstructed signal shape: {reconstructed_signal.shape}")

    # Compute metrics
    mse = mean_squared_error(original_signal, reconstructed_signal)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_signal, reconstructed_signal)
    r2 = r2_score(original_signal, reconstructed_signal)
    correlation, _ = pearsonr(original_signal, reconstructed_signal)
    original_fft = np.abs(np.fft.fft(original_signal))
    reconstructed_fft = np.abs(np.fft.fft(reconstructed_signal))
    spectral_mse = mean_squared_error(original_fft, reconstructed_fft)

    print(f"MSE between original and reconstructed: {mse:.6f}")
    print(f"RMSE between original and reconstructed: {rmse:.6f}")
    print(f"MAE between original and reconstructed: {mae:.6f}")
    print(f"R-squared score: {r2:.6f}")
    print(f"Pearson correlation coefficient: {correlation:.6f}")
    print(f"Spectral MSE (frequency domain): {spectral_mse:.6f}")

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

if __name__ == "__main__":
    main()