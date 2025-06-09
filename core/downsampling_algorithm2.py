import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np
import pywt

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

@keras.saving.register_keras_serializable()
class DownsampleTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, retention_rate, rate=0.3, **kwargs):
        super(DownsampleTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.retention_rate = retention_rate  
        if not 0 < retention_rate <= 1:
            raise ValueError("retention_rate must be between 0 and 1")
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
        self.bn.build(input_shape)
        self.residual_proj.build(input_shape)
        self.built = True

    def call(self, inputs, training=None):
        print(f"DownsampleTransformerBlock input shape: {inputs.shape}")
        tf.ensure_shape(inputs, [None, None, self.embed_dim])
        norm1 = self.layernorm1(inputs)
        attn_output, attn_weights = self.att(norm1, norm1, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        norm2 = self.layernorm2(out1)
        local_attn_output, local_attn_weights = self.local_att(norm2, norm2, return_attention_scores=True)
        local_attn_output = self.dropout2(local_attn_output, training=training)
        out2 = out1 + local_attn_output
        norm3 = self.layernorm3(out2)
        ffn_output = self.ffn(norm3)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = out2 + ffn_output
        
        # Derive importance scores from attention weights (hybrid scoring)
        print(f"Global attention weights shape: {attn_weights.shape}")
        tf.ensure_shape(attn_weights, [None, self.num_heads, None, None])
        print(f"Local attention weights shape: {local_attn_weights.shape}")
        tf.ensure_shape(local_attn_weights, [None, self.num_heads, None, None])
        
        # Global attention scores
        importance_scores_global = tf.reduce_mean(attn_weights, axis=1)  # Average across heads
        importance_scores_global = tf.reduce_mean(importance_scores_global, axis=1)  # Average across queries
        tf.ensure_shape(importance_scores_global, [None, None])
        
        # Local attention scores
        importance_scores_local = tf.reduce_mean(local_attn_weights, axis=1)  # Average across heads
        importance_scores_local = tf.reduce_mean(importance_scores_local, axis=1)  # Average across queries
        tf.ensure_shape(importance_scores_local, [None, None])
        
        # Weighted average for hybrid scoring
        importance_scores = 0.7 * importance_scores_global + 0.3 * importance_scores_local
        print(f"Combined importance scores shape: {importance_scores.shape}")
        tf.ensure_shape(importance_scores, [None, None])
        
        # Normalize with softmax
        importance_scores_squeezed = tf.nn.softmax(importance_scores, axis=-1)
        print(f"Importance scores normalized shape: {importance_scores_squeezed.shape}")
        tf.ensure_shape(importance_scores_squeezed, [None, None])
        
        # Dynamic retention rate calculation
        seq_len = tf.shape(out3)[1]
        reduced_seq_len = tf.cast(tf.round(tf.cast(seq_len, tf.float32) * self.retention_rate), tf.int32)
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
        if input_shape[1] is None:
            reduced_seq_len = None
        else:
            seq_len = input_shape[1]
            reduced_seq_len = max(1, int(round(seq_len * self.retention_rate)))
        return [(input_shape[0], reduced_seq_len, self.embed_dim), (input_shape[0], reduced_seq_len)]

    def get_config(self):
        config = super(DownsampleTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'retention_rate': self.retention_rate, 
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
        pos_encoding = self.get_sinusoidal_pos_encoding(self.maxlen, self.embed_dim)
        pos_encoding = tf.ensure_shape(pos_encoding, [1, self.maxlen, self.embed_dim])
        output = value_embeddings + pos_encoding
        output = tf.ensure_shape(output, [None, self.maxlen, self.embed_dim])
        print(f"TimeSeriesEmbedding output shape after positional encoding: {output.shape}")
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

def build_detail_transformer(input_seq_len, embed_dim, num_heads, ff_dim, num_transformer_blocks, retention_rate):
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
        layer_output = DownsampleTransformerBlock(embed_dim, num_heads, ff_dim, rate=0.3, retention_rate=retention_rate)(x)
        x = layer_output[0]
        all_indices.append(layer_output[1])
        print(f"After DownsampleTransformerBlock {i+1}: {x.shape}")
    output_seq_len = int(input_seq_len * (retention_rate ** num_transformer_blocks))
    output_seq_len = max(output_seq_len, 1)
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

def downsampling_loss(y_true, y_pred):
    mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
    y_true_fft = tf.abs(tf.signal.fft(tf.cast(y_true, tf.complex64)))
    y_pred_fft = tf.abs(tf.signal.fft(tf.cast(y_pred, tf.complex64)))
    freq_loss = tf.reduce_mean(tf.square(y_true_fft - y_pred_fft))
    return mse_loss + 0.5 * freq_loss