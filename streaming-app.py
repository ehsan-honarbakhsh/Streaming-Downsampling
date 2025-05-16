import tensorflow as tf
import keras
from keras import layers,regularizers
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os
import math
import json
import sys
#PyFlink libraries for real-time streaming
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink
from pyflink.common import Types
from pyflink.datastream.functions import FlatMapFunction
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.common.serialization import DeserializationSchema, SerializationSchema
from pyflink.java_gateway import get_gateway
from pyflink.datastream.connectors.kafka import DeliveryGuarantee



# Calculating the lengths of approximation (cA) and detail (cD) coefficients for a given signal length after applying discrete wavelet transform (DWT)
# Used to determine the expected sizes of wavelet coefficients, critical for configuring the model’s input and output shapes
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
        len_cA = len(coeffs[0])
        len_cD = len(coeffs[1])
    print(f"get_wavedec_coeff_lengths: signal_length={signal_length}, wavelet={wavelet.name}, level={level}, mode={mode}, len_cA={len_cA}, len_cD={len_cD}")
    return len_cA, len_cD

def load_ecg200(train_file_path, test_file_path):
    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        print(f"dataset files not found at: {train_file_path} or {test_file_path}")
        raise FileNotFoundError("Dataset files missing.")

    train_data = pd.read_csv(train_file_path, sep=r'\s+', header=None, engine='python')
    test_data = pd.read_csv(test_file_path, sep=r'\s+', header=None, engine='python')

    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values.astype(int)
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values.astype(int)

    #Converts labels from {-1, 1} to {0, 1} for binary classification compatibility
    y_train = np.where(y_train == -1, 0, 1)
    y_test = np.where(y_test == -1, 0, 1)

    return X_train, y_train, X_test, y_test

# Used in the build_detail_transformer function to process detail coefficients from wavelet decomposition, reducing sequence length while capturing temporal dependencies
class DownsampleTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2, **kwargs):
        super(DownsampleTransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
             layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.01))]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.conv = layers.Conv1D(
            filters=embed_dim, kernel_size=3, strides=2, padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        print(f"DownsampleTransformerBlock input shape: {inputs.shape}")
        #Applies layer normalization, multi-head attention, and dropout, adding the result to the input (residual connection)
        norm1 = self.layernorm1(inputs)
        attn_output = self.att(norm1, norm1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        #Applies layer normalization, FFN, and dropout, adding the result to the previous output (another residual connection).
        norm2 = self.layernorm2(out1)
        ffn_output = self.ffn(norm2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output
        #Passes the result through a Conv1D layer (downsamples by stride=2) and batch normalization
        conv_output = self.conv(out2)
        downsampled_output = self.bn(conv_output, training=training)
        print(f"After Conv1D and BN: {downsampled_output.shape}")
        return downsampled_output

    def get_config(self):
        config = super(DownsampleTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.ffn.layers[1].units,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

#Used in the build_detail_transformer function to prepare wavelet detail coefficients for transformer processing
class TimeSeriesEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(TimeSeriesEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.value_emb = layers.Dense(embed_dim)
    #Generates position encodings using sine and cosine functions
    #Ensures the model can distinguish the order of time steps
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
    
#Constructs a transformer-based model to process and downsample wavelet detail coefficients.
def build_detail_transformer(input_seq_len, embed_dim, num_heads, ff_dim, num_transformer_blocks=3):
    inputs = layers.Input(shape=(input_seq_len, 1))
    x = TimeSeriesEmbedding(maxlen=input_seq_len, embed_dim=embed_dim)(inputs)
    for _ in range(num_transformer_blocks):
        x = DownsampleTransformerBlock(embed_dim, num_heads, ff_dim, rate=0.2)(x)
    output_seq_len = input_seq_len // (2 ** num_transformer_blocks)
    if output_seq_len < 1:
        output_seq_len = 1
    print(f"build_detail_transformer: input_seq_len={input_seq_len}, output_seq_len={output_seq_len}")
    x = layers.Conv1D(filters=1, kernel_size=1, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(output_seq_len, kernel_regularizer=regularizers.l2(0.01))(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model

# Upsamples a compressed representation back to the original signal length using transposed convolutions
# Reconstructs the original signal from the combined downsampled approximation and detail coefficients
class LearnableUpsampler(keras.Model):
    def __init__(self, input_seq_len, target_original_length, initial_embed_dim=32, num_upsample_blocks_approx=None, **kwargs):
        super(LearnableUpsampler, self).__init__(**kwargs)
        self.input_seq_len = input_seq_len
        self.target_original_length = target_original_length
        self.initial_embed_dim = initial_embed_dim

        if num_upsample_blocks_approx is None:
            self.num_upsample_blocks_approx = max(1, math.ceil(math.log2(self.target_original_length / max(1, self.input_seq_len))))
        else:
            self.num_upsample_blocks_approx = num_upsample_blocks_approx

        self.initial_projection = layers.Conv1D(filters=self.initial_embed_dim, kernel_size=1, padding='same')

        self.upsample_layers = []
        current_filters = self.initial_embed_dim
        current_length = self.input_seq_len
        for i in range(self.num_upsample_blocks_approx):
            stride = 2
            kernel_size = 4
            output_filters = max(1, current_filters // 2)
            output_length = current_length * stride
            self.upsample_layers.append(
                layers.Conv1DTranspose(
                    filters=output_filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='same'
                )
            )
            self.upsample_layers.append(layers.BatchNormalization())
            self.upsample_layers.append(layers.ReLU())
            current_filters = output_filters
            current_length = output_length

        self.output_length = current_length
        self.output_filters = current_filters
        self.final_projection_dense = layers.Dense(self.target_original_length, activation=None)

    def call(self, inputs, training=None):
        print(f"LearnableUpsampler.call() input shape: {inputs.shape}")
        x = tf.expand_dims(inputs, axis=-1)
        x = self.initial_projection(x)
        for layer in self.upsample_layers:
            x = layer(x, training=training if isinstance(layer, layers.BatchNormalization) else None)
            print(f"After {layer.name}: {x.shape}")
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, self.output_length * self.output_filters])
        print(f"Shape after flattening: {x.shape}")
        reconstructed_signal = self.final_projection_dense(x)
        reconstructed_signal.set_shape([None, self.target_original_length])
        return reconstructed_signal

    def build(self, input_shape):
        initial_input_shape = (input_shape[0], input_shape[1], 1)
        self.initial_projection.build(initial_input_shape)
        current_shape = [input_shape[0], input_shape[1], self.initial_embed_dim]
        for layer in self.upsample_layers:
            if not layer.built:
                if isinstance(layer, layers.Conv1DTranspose):
                    current_length = current_shape[1]
                    if current_length is not None:
                        new_length = current_length * layer.strides[0]
                    else:
                        new_length = None
                    new_filters = layer.filters
                    current_shape = [current_shape[0], new_length, new_filters]
                elif isinstance(layer, layers.BatchNormalization):
                    layer.build(current_shape)
                else:
                    layer.build(current_shape)
        flattened_dim = self.output_length * self.output_filters
        self.final_projection_dense.build((input_shape[0], flattened_dim))
        super(LearnableUpsampler, self).build(input_shape)

    def get_config(self):
        config = super(LearnableUpsampler, self).get_config()
        config.update({
            'input_seq_len': self.input_seq_len,
            'target_original_length': self.target_original_length,
            'initial_embed_dim': self.initial_embed_dim,
            'num_upsample_blocks_approx': self.num_upsample_blocks_approx
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


#Combines wavelet decomposition, transformer-based downsampling of detail coefficients, and learnable upsampling to reconstruct ECG signals from a compressed representation
class WaveletReconstructionModel(keras.Model):
    def __init__(self, detail_transformer_model, wavelet_name, approx_ds_factor,
                 original_length, signal_coeffs_len, dwt_level=1, normalize_details=False,
                 upsampler_initial_embed_dim=32, decomposition_mode='symmetric', **kwargs):
        super(WaveletReconstructionModel, self).__init__(**kwargs)
        self.detail_transformer = detail_transformer_model
        self.wavelet_name = wavelet_name
        self.approx_ds_factor = approx_ds_factor
        self.original_length = original_length
        self.signal_coeffs_len = signal_coeffs_len
        self.dwt_level = dwt_level
        self.normalize_details = normalize_details
        self.decomposition_mode = decomposition_mode
        self.upsampler_initial_embed_dim = upsampler_initial_embed_dim

        if self.normalize_details:
            self.detail_norm_layer = layers.LayerNormalization(epsilon=1e-6)

        self.detail_ds_len = None
        self.learnable_upsampler = None

    def build(self, input_shape):
        print(f"Building WaveletReconstructionModel with input_shape: {input_shape}")
        if not self.detail_transformer.built:
            dummy_detail_input_shape = tf.TensorShape((None, self.signal_coeffs_len, 1))
            self.detail_transformer.build(dummy_detail_input_shape)
            print("Built detail_transformer within WaveletReconstructionModel build.")
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
        combined_ds_len = actual_approx_ds_len + self.detail_ds_len
        print(f"Actual combined downsampled length: {combined_ds_len}")
        if self.learnable_upsampler is None:
            self.learnable_upsampler = LearnableUpsampler(
                input_seq_len=combined_ds_len,
                target_original_length=self.original_length,
                initial_embed_dim=self.upsampler_initial_embed_dim
            )
            self.learnable_upsampler.build(tf.TensorShape((None, combined_ds_len)))
            print("Built LearnableUpsampler:")
            self.learnable_upsampler.summary(line_length=100)
        super(WaveletReconstructionModel, self).build(input_shape)

    def call(self, original_signals_batch, training=None, **kwargs):
        print(f"Calling WaveletReconstructionModel with input shape: {original_signals_batch.shape}")
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
        if self.normalize_details:
            detail_coeffs_reshaped = self.detail_norm_layer(detail_coeffs_reshaped, training=training)
        detail_downsampled = self.detail_transformer(detail_coeffs_reshaped, training=training)
        print(f"detail_downsampled shape: {detail_downsampled.shape}")
        combined_downsampled = tf.concat([approx_downsampled, detail_downsampled], axis=1)
        print(f"combined_downsampled shape: {combined_downsampled.shape}")
        reconstructed_signal = self.learnable_upsampler(combined_downsampled, training=training)
        reconstructed_signal.set_shape([None, self.original_length])
        return reconstructed_signal

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.original_length)

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

    #returns the compressed representation (concatenated downsampled approximation and detail coefficients) without upsampling
    def get_downsampled_representation(self, original_signals_batch):
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
                approx_coeffs_reshaped, ksize=self.approx_ds_factor,
                strides=self.approx_ds_factor, padding='VALID')
            approx_downsampled = tf.squeeze(approx_downsampled_pooled, axis=-1)
        else:
            approx_downsampled = approx_coeffs
        detail_coeffs_reshaped = tf.expand_dims(detail_coeffs, axis=-1)
        if self.normalize_details:
            detail_coeffs_reshaped = self.detail_norm_layer(detail_coeffs_reshaped, training=False)
        detail_downsampled = self.detail_transformer(detail_coeffs_reshaped, training=False)
        combined_downsampled = tf.concat([approx_downsampled, detail_downsampled], axis=1)
        return combined_downsampled

    def get_config(self):
        config = super(WaveletReconstructionModel, self).get_config()
        config.update({
            'wavelet_name': self.wavelet_name,
            'approx_ds_factor': self.approx_ds_factor,
            'original_length': self.original_length,
            'signal_coeffs_len': self.signal_coeffs_len,
            'dwt_level': self.dwt_level,
            'normalize_details': self.normalize_details,
            'upsampler_initial_embed_dim': self.upsampler_initial_embed_dim,
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
            upsampler_initial_embed_dim=config['upsampler_initial_embed_dim'],
            decomposition_mode=config['decomposition_mode']
        )
        return instance


class ECGDeserializationSchema(DeserializationSchema):
    def __init__(self):
        super(ECGDeserializationSchema, self).__init__()
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

class ECGSerializationSchema(SerializationSchema):
    def __init__(self, topic='ecg-output'):
        super(ECGSerializationSchema, self).__init__()
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

#Setting up a real-time streaming pipeline using Apache Flink to process ECG signals with the trained model.
#Enables real-time ECG signal reconstruction, suitable for applications like remote monitoring
def stream_processing(data_mean, data_std):
    custom_objects = {
        'DownsampleTransformerBlock': DownsampleTransformerBlock,
        'TimeSeriesEmbedding': TimeSeriesEmbedding,
        'LearnableUpsampler': LearnableUpsampler,
        'WaveletReconstructionModel': WaveletReconstructionModel
    }

    try:
        reconstruction_model_loaded = keras.models.load_model(
            'reconstruction_model.keras',
            custom_objects=custom_objects,
            compile=False
        )
        detail_transformer_loaded = keras.models.load_model(
            'detail_transformer_model.keras',
            custom_objects=custom_objects
        )
        learnable_upsampler_loaded = keras.models.load_model(
            'learnable_upsampler_model.keras',
            custom_objects=custom_objects
        )
        reconstruction_model_loaded.detail_transformer = detail_transformer_loaded
        reconstruction_model_loaded.learnable_upsampler = learnable_upsampler_loaded
        reconstruction_model = reconstruction_model_loaded
    except Exception as e:
        print(f"Error loading models for streaming: {e}")
        sys.exit(1)

    #normalizes input signals, processes them through the model, and denormalizes the output.
    class ECGProcessor(FlatMapFunction):
        def __init__(self, model, mean, std):
            self.model = model
            self.data_mean = mean
            self.data_std = std
            self.wavelet_name = 'db4'
            self.dwt_level = 1
            self.decomposition_mode = 'symmetric'
            self.approx_ds_factor = 2
            self.normalize_details = True

        def flat_map(self, signal):
            if signal is None:
                print("Received null signal")
                return
            try:
                print(f"Processing signal: {signal[:10]}... (length: {len(signal)})")
                signal_np = np.array(signal, dtype=np.float32)
                original_length = self.model.original_length
                if len(signal_np) != original_length:
                    print(f"Warning: Signal length {len(signal_np)} does not match model expected length {original_length}. Skipping.")
                    return
                signal_normalized = (signal_np - self.data_mean) / self.data_std
                input_tensor = tf.constant(signal_normalized, dtype=tf.float32)
                input_tensor = tf.expand_dims(input_tensor, axis=0)
                reconstructed_signal_tensor = self.model(input_tensor, training=False)
                reconstructed_signal_np = tf.squeeze(reconstructed_signal_tensor, axis=0).numpy()
                reconstructed_signal_denorm = reconstructed_signal_np * self.data_std + self.data_mean
                output = reconstructed_signal_denorm.tolist()
                print(f"Processed signal output: {output[:10]}... (length: {len(output)})")
                yield output
            except Exception as e:
                print(f"Error processing signal {signal[:10]}...: {e}")
                pass

    env = StreamExecutionEnvironment.get_execution_environment()
    kafka_jar_path = "file:///Users/ehsanhonarbakhsh/Documents/GitHub/Downsampling/flink-connector-kafka-4.0.0-2.0.jar"
    kafka_clients_jar_path = "file:///Users/ehsanhonarbakhsh/Documents/GitHub/Downsampling/kafka-clients-4.0.0.jar"
    all_jars = [kafka_jar_path, kafka_clients_jar_path]

    print("Adding JARs to Flink environment:")
    for jar_path_full in all_jars:
        local_path = jar_path_full.replace("file://", "")
        if not os.path.exists(local_path):
            print(f"Error: JAR not found at local path: {local_path}")
            sys.exit(1)
        env.add_jars(jar_path_full)
        print(f"- Added: {jar_path_full}")

    print("Configuring Kafka source with topic 'ecg-input'...")
    deserializer = ECGDeserializationSchema()
    source = KafkaSource.builder() \
        .set_bootstrap_servers('localhost:9092') \
        .set_topics('ecg-input') \
        .set_group_id('flink-ecg-group') \
        .set_property('auto.offset.reset', 'latest') \
        .set_value_only_deserializer(deserializer) \
        .build()

    print("Configuring Kafka sink with topic 'ecg-output'...")
    serializer = ECGSerializationSchema(topic='ecg-output')
    sink = KafkaSink.builder() \
        .set_bootstrap_servers('localhost:9092') \
        .set_record_serializer(serializer) \
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE) \
        .build()

    stream = env.from_source(source, WatermarkStrategy.no_watermarks(), "Kafka Source")
    ecg_processor_instance = ECGProcessor(reconstruction_model, data_mean, data_std)
    processed_stream = stream.flat_map(ecg_processor_instance, output_type=Types.LIST(Types.FLOAT()))
    processed_stream.sink_to(sink)

    print("\nStarting Flink stream processing job...")
    env.execute("ECG Stream Processing")

#Defines a custom loss function combining MSE, frequency-domain loss, and perceptual loss for training the reconstruction model

#A convolutional network to extract high-level features from signals, used in perceptual_loss
_perceptual_model = None
def get_perceptual_model(input_shape=(96, 1)):
    global _perceptual_model
    if _perceptual_model is None:
        _perceptual_model = keras.Sequential([
            layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape),
            layers.Conv1D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(256, 3, padding='same', activation='relu'),
        ])
    return _perceptual_model
#Computes the mean squared difference between features extracted from true and predicted signals
def perceptual_loss(y_true, y_pred):
    perceptual_model = get_perceptual_model()
    y_true_features = perceptual_model(y_true)
    y_pred_features = perceptual_model(y_pred)
    return tf.reduce_mean(tf.square(y_true_features - y_pred_features))

#Computes the mean absolute difference between the FFTs of true and predicted signals, ensuring frequency-domain similarity
def frequency_loss(y_true, y_pred):
    y_true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
    y_pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))
    return tf.reduce_mean(tf.abs(y_true_fft - y_pred_fft))

#Combines MSE (time-domain), frequency loss (0.1 weight), and perceptual loss (0.05 weight) for a balanced objective
def custom_loss(y_true, y_pred):
    mse = keras.losses.mean_squared_error(y_true, y_pred)
    freq_loss = frequency_loss(y_true, y_pred)
    perc_loss = perceptual_loss(y_true, y_pred)
    return mse + 0.1 * freq_loss + 0.05 * perc_loss

#Enhances the training dataset to improve model robustness.
#Adds Gaussian noise to the input signals.
def augment_data(X, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, X.shape)
    X_augmented = X + noise
    return X_augmented
#Applies mixup augmentation, blending pairs of signals with weights drawn from a Beta distribution.
def mixup_data(X, y, alpha=0.2):
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D inputs, got X with {X.ndim} dims and y with {y.ndim} dims")
    batch_size = X.shape[0]
    indices = np.random.permutation(batch_size)
    lam = np.random.beta(alpha, alpha, batch_size)
    X_mixed = lam[:, None] * X + (1 - lam[:, None]) * X[indices]
    y_mixed = lam[:, None] * y + (1 - lam[:, None]) * y[indices]
    return X_mixed, y_mixed


#Orchestrates data loading, model training, evaluation, visualization, and streaming pipeline execution.
def main():
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    base_path = "."
    train_file = os.path.join(base_path, "ECG200_TRAIN.txt")
    test_file = os.path.join(base_path, "ECG200_TEST.txt")

    embed_dim = 64
    num_heads = 8
    ff_dim = 64
    num_transformer_blocks = 4
    wavelet_name = 'db4'
    dwt_level = 1
    approx_ds_factor = 2
    original_length = 96
    len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(original_length, wavelet_name, dwt_level, 'symmetric')
    print(f"Computed signal_coeffs_len: {signal_coeffs_len}")
    normalize_details = True
    upsampler_initial_embed_dim = 32
    decomposition_mode = 'symmetric'
    batch_size = 128
    epochs = 250
    learning_rate = 0.0001

    X_train, y_train, X_test, y_test = load_ecg200(train_file, test_file)
    data_mean = np.mean(X_train, axis=0)
    data_std = np.std(X_train, axis=0)
    X_train_normalized = (X_train - data_mean) / data_std
    X_test_normalized = (X_test - data_mean) / data_std

    # Data augmentation
    X_train_augmented = augment_data(X_train_normalized)
    X_train_mixed, y_train_mixed = mixup_data(X_train_normalized, X_train_normalized)
    print(f"X_train_normalized shape: {X_train_normalized.shape}")
    print(f"X_train_augmented shape: {X_train_augmented.shape}")
    print(f"X_train_mixed shape: {X_train_mixed.shape}")
    X_train_combined = np.concatenate([X_train_normalized, X_train_augmented, X_train_mixed], axis=0)
    y_train_combined = np.concatenate([X_train_normalized, X_train_augmented, y_train_mixed], axis=0)

    detail_transformer = build_detail_transformer(signal_coeffs_len, embed_dim, num_heads, ff_dim, num_transformer_blocks)
    reconstruction_model = WaveletReconstructionModel(
        detail_transformer_model=detail_transformer,
        wavelet_name=wavelet_name,
        approx_ds_factor=approx_ds_factor,
        original_length=original_length,
        signal_coeffs_len=signal_coeffs_len,
        dwt_level=dwt_level,
        normalize_details=normalize_details,
        upsampler_initial_embed_dim=upsampler_initial_embed_dim,
        decomposition_mode=decomposition_mode
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    print(f"Optimizer type before compile: {type(optimizer)}")
    reconstruction_model.compile(optimizer=optimizer, loss=custom_loss)
    print(f"Optimizer type after compile: {type(reconstruction_model.optimizer)}")
    print("Wavelet Reconstruction Model Summary:")
    reconstruction_model.build(input_shape=(None, original_length))
    reconstruction_model.summary(line_length=150)

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    print("--- Training Wavelet Reconstruction Model for 150 epochs ---")
    history = reconstruction_model.fit(
        X_train_combined, X_train_combined,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_normalized, X_test_normalized),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    print("\n--- Evaluating Model ---")
    mse_train = mean_squared_error(X_train_normalized, reconstruction_model.predict(X_train_normalized))
    mse_test = mean_squared_error(X_test_normalized, reconstruction_model.predict(X_test_normalized))
    print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")

    print("\n--- Downsampled Representation Example ---")
    sample_input = X_test_normalized[:10]
    downsampled_representation = reconstruction_model.get_downsampled_representation(sample_input)
    print("Downsampled Representation Shape:", downsampled_representation.shape)

    print("\n--- Saving Models for Streaming ---")
    reconstruction_model.save('reconstruction_model.keras')
    detail_transformer.save('detail_transformer_model.keras')
    reconstruction_model.learnable_upsampler.save('learnable_upsampler_model.keras')

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(X_test_normalized[0], label='Original Signal', alpha=0.7)
    plt.plot(reconstruction_model.predict(X_test_normalized)[0], label='Reconstructed Signal', alpha=0.7)
    plt.title(f'Original vs. Reconstructed Signal\n(96 Samples Each)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 2, 3)
    downsampled_signal = reconstruction_model.get_downsampled_representation(X_test_normalized[:1])[0].numpy()
    num_downsampled_samples = len(downsampled_signal)
    plt.plot(downsampled_signal, label='Downsampled Signal', alpha=0.7, marker='o')
    plt.title(f'Downsampled Time Series\n({num_downsampled_samples} Samples)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(X_test_normalized[0], label='Original Signal', alpha=0.7, color='orange')
    plt.title(f'Original Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return data_mean, data_std

if __name__ == "__main__":
    data_mean, data_std = main()
    stream_processing(data_mean, data_std)