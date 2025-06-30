import logging
import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np
import pywt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Calculating the lengths of approximation (cA) and detail (cD) coefficients
def get_wavedec_coeff_lengths(signal_length, wavelet, level, mode='symmetric'):
    """Compute lengths of approximation and detail coefficients for wavelet decomposition.
    
    Args:
        signal_length (int): Length of the input signal.
        wavelet (str or pywt.Wavelet): Wavelet name (e.g., 'db4') or Wavelet object.
        level (int): Decomposition level (non-negative integer).
        mode (str): Signal extension mode (default: 'symmetric').
    
    Returns:
        tuple: (len_cA, len_cD) - lengths of approximation and detail coefficients.
    
    Raises:
        ValueError: If level is negative or wavelet is invalid.
    """
    if isinstance(wavelet, str):
        try:
            wavelet = pywt.Wavelet(wavelet)
        except ValueError as e:
            raise ValueError(f"Invalid wavelet name: {wavelet}") from e
    
    if not isinstance(wavelet, pywt.Wavelet):
        raise ValueError(f"Expected pywt.Wavelet object, got {type(wavelet)}")
    
    if level < 0:
        raise ValueError(f"Decomposition level must be >= 0, got {level}")
    
    len_cA = signal_length
    len_cD = 0
    for _ in range(level):
        len_cA = pywt.dwt_coeff_len(len_cA, wavelet.dec_len, mode)
        len_cD += len_cA
    if level == 0:
        len_cA = pywt.dwt_coeff_len(signal_length, wavelet.dec_len, mode)
        len_cD = len_cA
    logger.debug(f"get_wavedec_coeff_lengths: signal_length={signal_length}, wavelet={wavelet.name}, level={level}, mode={mode}, len_cA={len_cA}, len_cD={len_cD}")
    return len_cA, len_cD

@keras.saving.register_keras_serializable()
class DownsampleTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, min_retention_rate=0.1, target_cumulative_importance=0.99, rate=0.3, **kwargs):
        """Initialize a transformer block with dynamic downsampling based on importance scores.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward network dimension.
            min_retention_rate (float): Minimum fraction of points to retain (default: 0.1).
            target_cumulative_importance (float): Target cumulative importance score (default: 0.99).
            rate (float): Dropout rate (default: 0.3).
        """
        super(DownsampleTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.min_retention_rate = min_retention_rate
        self.target_cumulative_importance = target_cumulative_importance
        if not 0 < min_retention_rate <= 1:
            raise ValueError("min_retention_rate must be between 0 and 1")
        if not 0 < target_cumulative_importance <= 1:
            raise ValueError("target_cumulative_importance must be between 0 and 1")
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
        logger.debug(f"DownsampleTransformerBlock input shape: {inputs.shape}")
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
        logger.debug(f"Global attention weights shape: {attn_weights.shape}")
        tf.ensure_shape(attn_weights, [None, self.num_heads, None, None])
        logger.debug(f"Local attention weights shape: {local_attn_weights.shape}")
        tf.ensure_shape(local_attn_weights, [None, self.num_heads, None, None])
        
        # Global attention scores
        importance_scores_global = tf.reduce_mean(attn_weights, axis=1)
        importance_scores_global = tf.reduce_mean(importance_scores_global, axis=1)
        tf.ensure_shape(importance_scores_global, [None, None])
        
        # Local attention scores
        importance_scores_local = tf.reduce_mean(local_attn_weights, axis=1)
        importance_scores_local = tf.reduce_mean(importance_scores_local, axis=1)
        tf.ensure_shape(importance_scores_local, [None, None])
        
        # Weighted average for hybrid scoring
        importance_scores = 0.7 * importance_scores_global + 0.3 * importance_scores_local
        logger.debug(f"Combined importance scores shape: {importance_scores.shape}")
        tf.ensure_shape(importance_scores, [None, None])
        
        # Normalize importance scores to sum to 1
        importance_scores_normalized = importance_scores / (tf.reduce_sum(importance_scores, axis=-1, keepdims=True) + 1e-10)
        logger.debug(f"Importance scores normalized shape: {importance_scores_normalized.shape}")
        
        # Sort importance scores and compute cumulative sum
        sorted_importance, sorted_indices = tf.math.top_k(importance_scores_normalized, k=tf.shape(importance_scores_normalized)[-1])
        cumulative_importance = tf.cumsum(sorted_importance, axis=-1)
        
        # Find number of points to reach target_cumulative_importance
        mask = tf.cast(cumulative_importance <= self.target_cumulative_importance, tf.float32)
        num_points = tf.reduce_sum(mask, axis=-1, keepdims=True)
        min_points = tf.cast(tf.round(tf.cast(tf.shape(inputs)[1], tf.float32) * self.min_retention_rate), tf.int32)
        num_points = tf.maximum(tf.cast(num_points, tf.int32), min_points)
        num_points = tf.minimum(num_points, tf.shape(inputs)[1])
        
        # Select top_k indices dynamically per sample
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        top_k_indices = []
        for i in range(batch_size):
            k = num_points[i, 0]
            k = tf.minimum(k, seq_len)
            indices = sorted_indices[i, :k]
            top_k_indices.append(indices)
        top_k_indices = tf.stack(top_k_indices, axis=0)
        top_k_indices = tf.sort(top_k_indices, axis=-1)
        logger.debug(f"Top k indices shape: {top_k_indices.shape}")
        
        # Downsample output
        downsampled_output = tf.gather(out3, top_k_indices, batch_dims=1, axis=1)
        logger.debug(f"Downsampled output shape: {downsampled_output.shape}")
        tf.ensure_shape(downsampled_output, [None, None, self.embed_dim])
        residual = self.residual_proj(inputs)
        residual_downsampled = tf.gather(residual, top_k_indices, batch_dims=1, axis=1)
        logger.debug(f"Residual downsampled shape: {residual_downsampled.shape}")
        downsampled_output = downsampled_output + residual_downsampled
        downsampled_output = self.bn(downsampled_output, training=training)
        logger.debug(f"After attention-based downsampling and BN: {downsampled_output.shape}")
        return downsampled_output, top_k_indices

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], None, self.embed_dim), (input_shape[0], None)]

    def get_config(self):
        config = super(DownsampleTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'min_retention_rate': self.min_retention_rate,
            'target_cumulative_importance': self.target_cumulative_importance,
        })
        return config

@keras.saving.register_keras_serializable()
class TimeSeriesEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        """Embed time series data with positional encoding.
        
        Args:
            maxlen (int): Maximum sequence length.
            embed_dim (int): Embedding dimension.
        """
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
        logger.debug(f"TimeSeriesEmbedding build input shape: {input_shape}")
        super(TimeSeriesEmbedding, self).build(input_shape)

    def get_sinusoidal_pos_encoding(self, maxlen, embed_dim):
        position = tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / embed_dim))
        pos_encoding = position * div_term
        pos_encoding = tf.concat([tf.sin(pos_encoding), tf.cos(pos_encoding)], axis=-1)
        pos_encoding = pos_encoding[:, :embed_dim]
        return pos_encoding[tf.newaxis, :, :]

    def call(self, x):
        logger.debug(f"TimeSeriesEmbedding input shape: {x.shape}")
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)
        x = tf.ensure_shape(x, [None, self.maxlen, 1])
        value_embeddings = x @ tf.expand_dims(self.kernel, axis=0)
        value_embeddings = value_embeddings + self.bias
        tf.ensure_shape(value_embeddings, [None, self.maxlen, self.embed_dim])
        logger.debug(f"Value embeddings shape: {value_embeddings.shape}")
        pos_encoding = self.get_sinusoidal_pos_encoding(self.maxlen, self.embed_dim)
        tf.ensure_shape(pos_encoding, [1, self.maxlen, self.embed_dim])
        output = value_embeddings + pos_encoding
        tf.ensure_shape(output, [None, self.maxlen, self.embed_dim])
        logger.debug(f"TimeSeriesEmbedding output shape after positional encoding: {output.shape}")
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

def build_detail_transformer(input_seq_len, embed_dim, num_heads, ff_dim, num_transformer_blocks, min_retention_rate=0.1, target_cumulative_importance=0.99):
    """Build a transformer model for detail coefficient processing.
    
    Args:
        input_seq_len (int): Input sequence length.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward network dimension.
        num_transformer_blocks (int): Number of transformer blocks.
        min_retention_rate (float): Minimum retention rate (default: 0.1).
        target_cumulative_importance (float): Target cumulative importance (default: 0.99).
    
    Returns:
        keras.Model: Transformer model.
    """
    inputs = layers.Input(shape=(input_seq_len, 1))
    x = TimeSeriesEmbedding(maxlen=input_seq_len, embed_dim=embed_dim)(inputs)
    logger.debug(f"After TimeSeriesEmbedding: {x.shape}")
    if len(x.shape) != 3 or x.shape[1] != input_seq_len or x.shape[2] != embed_dim:
        raise ValueError(
            f"Expected TimeSeriesEmbedding output shape (batch_size, {input_seq_len}, {embed_dim}), "
            f"got {x.shape}"
        )
    all_indices = []
    current_seq_len = input_seq_len
    for i in range(num_transformer_blocks):
        logger.debug(f"Before DownsampleTransformerBlock {i+1}: {x.shape}")
        layer_output = DownsampleTransformerBlock(
            embed_dim, num_heads, ff_dim, rate=0.3,
            min_retention_rate=min_retention_rate,
            target_cumulative_importance=target_cumulative_importance
        )(x)
        x = layer_output[0]
        all_indices.append(layer_output[1])
        current_seq_len = max(1, int(current_seq_len * min_retention_rate))
        logger.debug(f"After DownsampleTransformerBlock {i+1}: {x.shape}")
    x = layers.Conv1D(filters=1, kernel_size=1, padding='same')(x)
    logger.debug(f"After Conv1D: {x.shape}")
    x = layers.Flatten()(x)
    logger.debug(f"After Flatten: {x.shape}")
    # Estimate input units to Dense layer based on min_retention_rate
    input_units = max(1, int(input_seq_len * (min_retention_rate ** num_transformer_blocks)))
    # Reshape to ensure fixed input shape to Dense layer
    x = layers.Reshape((input_units,))(x)
    logger.debug(f"After Reshape: {x.shape}")
    output_units = input_units  # Match output to input for consistency
    x = layers.Dense(output_units, kernel_regularizer=regularizers.l2(0.03))(x)
    logger.debug(f"After Dense: {x.shape}")
    model = keras.Model(inputs=inputs, outputs=[x] + all_indices)
    return model

@keras.saving.register_keras_serializable()
class WaveletDownsamplingModel(keras.Model):
    def __init__(self, detail_transformer_model, wavelet_name, approx_ds_factor,
                 original_length, signal_coeffs_len, dwt_level=1, normalize_details=False,
                 decomposition_mode='symmetric', min_retention_rate=0.1, **kwargs):
        """Initialize a wavelet-based downsampling model.
        
        Args:
            detail_transformer_model (keras.Model): Transformer model for detail coefficients.
            wavelet_name (str): Wavelet name (e.g., 'db4').
            approx_ds_factor (int): Downsampling factor for approximation coefficients.
            original_length (int): Original signal length.
            signal_coeffs_len (int): Length of detail coefficients.
            dwt_level (int): Wavelet decomposition level (default: 1).
            normalize_details (bool): Whether to normalize detail coefficients (default: False).
            decomposition_mode (str): Wavelet decomposition mode (default: 'symmetric').
            min_retention_rate (float): Minimum retention rate (default: 0.1).
        """
        super(WaveletDownsamplingModel, self).__init__(**kwargs)
        self.detail_transformer = detail_transformer_model
        self.wavelet_name = wavelet_name
        self.approx_ds_factor = approx_ds_factor
        self.original_length = original_length
        self.signal_coeffs_len = signal_coeffs_len
        self.dwt_level = dwt_level
        self.normalize_details = normalize_details
        self.decomposition_mode = decomposition_mode
        self.min_retention_rate = min_retention_rate
        self.detail_ds_len = None

        if self.normalize_details:
            self.detail_norm_layer = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        logger.debug(f"Building WaveletDownsamplingModel with input_shape: {input_shape}")
        if not self.detail_transformer.built:
            dummy_detail_input_shape = tf.TensorShape((None, self.signal_coeffs_len, 1))
            self.detail_transformer.build(dummy_detail_input_shape)
            logger.debug("Built detail_transformer within WaveletDownsamplingModel build.")
        dummy_input = tf.zeros((1, self.signal_coeffs_len, 1))
        output = self.detail_transformer(dummy_input)
        self.detail_ds_len = max(1, int(self.signal_coeffs_len * self.min_retention_rate))
        logger.debug(f"Estimated detail transformer output length: {self.detail_ds_len}")
        if self.detail_ds_len is None:
            raise ValueError("detail_transformer output length is None.")
        len_cA, len_cD = get_wavedec_coeff_lengths(
            self.original_length, self.wavelet_name, self.dwt_level, self.decomposition_mode
        )
        logger.debug(f"Wavelet coefficients: len_cA={len_cA}, len_cD={len_cD}")
        if self.approx_ds_factor > 1:
            actual_approx_len = (len_cA - 1) // self.approx_ds_factor + 1
        else:
            actual_approx_len = len_cA
        logger.debug(f"actual_approx_len={actual_approx_len}, detail_ds_len={self.detail_ds_len}")
        self.combined_ds_length = actual_approx_len + self.detail_ds_len
        logger.debug(f"Actual combined downsampled length: {self.combined_ds_length}")
        super(WaveletDownsamplingModel, self).build(input_shape)

    def call(self, inputs, training=None, return_indices=False, **kwargs):
        logger.debug(f"Calling WaveletDownsamplingModel with input shape: {inputs.shape}")
        if len(inputs.shape) == 3 and inputs.shape[-1] == 1:
            inputs_for_model = tf.squeeze(inputs, axis=-1)
        else:
            inputs_for_model = inputs
        approx_coeffs, detail_coeffs = tf.py_function(
            func=self._model_batch,
            inp=[inputs_for_model],
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
                padding='valid'
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
        logger.debug(f"approx_downsampled shape: {approx_downsampled.shape}")
        detail_coeffs_reshaped = tf.expand_dims(detail_coeffs, axis=-1)
        logger.debug(f"detail_coeffs shape: {detail_coeffs_reshaped.shape}")
        if self.detail_normalization:
            norm_layer = getattr(self, 'detail_norm_layer', None)
            if norm_layer is None:
                raise ValueError("detail_norm_layer is None but detail_normalization is True")
            detail_coeffs_reshaped = norm_layer(detail_coeffs_reshaped, training=training)
        logger.debug(f"detail_transformer type: {type(self.detail_transformer)}")
        logger.debug(f"detail_transformer callable: {callable(self.detail_transformer)}")
        if self.detail_transformer is None:
            raise ValueError("detail_transformer is None")
        if not callable(self.detail_transformer):
            raise ValueError(f"detail_transformer is not callable: {type(self.detail_transformer)}")
        detail_outputs = self.detail_transformer(detail_coeffs_reshaped, training=training)
        detail_downsampled = detail_outputs[0]
        detail_indices_list = detail_outputs[1:]
        logger.debug(f"detail_downsampled shape: {detail_downsampled.shape}")
        combined_downsampled = tf.concat([approx_downsampled, detail_downsampled], axis=1)
        logger.debug(f"combined_downsampled shape: {combined_downsampled.shape}")
        if return_indices:
            return combined_downsampled, [approx_indices] + detail_indices_list
        return combined_downsampled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], None)

    def _model_batch(self, batch_tensor):
        batch_np = batch_tensor.numpy()
        if len(batch_np.shape) == 3 and batch_np.shape[-1] == 1:
            batch_np = np.squeeze(batch_np, axis=-1)
        approx_coeffs_list = []
        detail_coeffs_list = []
        for row in batch_np:
            if len(row.shape) > 1:
                row = np.squeeze(row)
            if self.dwt_level == 0:
                norm, detail = pywt.dwt(row, self.wavelet_name, mode=self.decomposition_mode)
            else:
                coeffs = pywt.wavedec(row, self.wavelet_name, level=self.dwt_level, mode=self.decomposition_mode)
                norm = coeffs[0]
                detail = coeffs[1]
            approx_coeffs_list.append(norm)
            detail_coeffs_list.append(detail)
        return np.array(approx_coeffs_list, dtype=np.float32), np.array(detail_coeffs_list, dtype=np.float32)

    def get_config(self):
        config = super(DictWaveletTransform, self).get_config()
        config.update({
            'detail_transformer_model': keras.utils.serialize_keras_object(self.transformer),
            'wavelet_name': self.wavelet_name,
            'approx_ds_factor': self.approx_ds_factor,
            'original_length': self.original_length,
            'signal_coeffs_len': self.signal_coeffs_len,
            'dwt_level': self.dwt_level,
            'normalize_details': self.detail_normalization,
            'decomposition_mode': self.decomposition_mode,
            'min_retention_rate': self.min_retention_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        transformer_config = config.pop('transformer')
        transformer = keras.utils.deserialize_keras_object(transformer_config)
        instance = cls(
            detail_transformer=transformer,
            wavelet_name=config['wavelet_name'],
            approx_signal_factor=config['approx_signal_factor'],
            original_length=config['original_length'],
            signal_coeffs_len=config['signal_coeffs_len'],
            dwt_level=config['dwt_level'],
            normalize_details=config['detail_normalization'],
            decomposition_mode=config['decomposition_mode'],
            min_retention_rate=config['min_retention_rate']
        )
        return instance

def downsampling_loss(input_true, target_pred):
    """Custom loss function combining MSE and frequency-domain loss.
    
    Args:
        input_true (tf.Tensor): Input true values.
        target_pred (tf.Tensor): Target predicted values.
    
    Returns:
        tf.Tensor: Combined loss value.
    """
    mse_loss = keras.losses.mean_squared_error(input_true, target_pred)
    input_true_fft = tf.abs(tf.signal.fft(tf.cast(input_true, tf.complex64)))
    target_pred_fft = tf.abs(tf.signal.fft(tf.cast(target_pred, tf.complex64)))
    freq_loss = tf.reduce_mean(tf.square(input_true_fft - target_pred_fft))
    return mse_loss + 0.5 * freq_loss