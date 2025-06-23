import unittest
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from unittest.mock import patch, MagicMock
import json
import logging
import psutil
import gc
from core.downsampling_algorithm2 import (
    get_wavedec_coeff_lengths,
    TimeSeriesEmbedding,
    DownsampleTransformerBlock,
    WaveletDownsamplingModel,
    build_detail_transformer,
    downsampling_loss
)
from core.streaming_pipeline import  DeserializationSchema, SerializationSchema

# Suppress TensorFlow warnings and set deterministic behavior
tf.keras.utils.set_random_seed(42)
np.random.seed(42)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class TestDownsampling(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.signal_length = 128
        self.wavelet_name = 'db4'
        self.dwt_level = 1
        self.embed_dim = 32
        self.num_heads = 4
        self.ff_dim = 32
        self.retention_rate = 0.8
        self.maxlen = 64
        self.batch_size = 2
        self.original_length = 150
        tf.keras.backend.clear_session()
        self.config = {
            'chunk_size': 20,
            'chunk_size_fraction': 0.1,
            'min_chunk_size': 10,
            'max_chunk_size': 100
        }

    def tearDown(self):
        """Clean up after each test."""
        tf.keras.backend.clear_session()
        gc.collect()

    def test_get_wavedec_coeff_lengths(self):
        """Test get_wavedec_coeff_lengths function."""
        len_cA, len_cD = get_wavedec_coeff_lengths(self.signal_length, self.wavelet_name, self.dwt_level, mode='symmetric')
        self.assertGreater(len_cA, 0, "Approximation coefficient length should be positive")
        self.assertGreater(len_cD, 0, "Detail coefficient length should be positive")
        self.assertAlmostEqual(len_cA, len_cD, msg="Expected similar lengths for db4 wavelet")

        # Test edge case: level=0
        len_cA, len_cD = get_wavedec_coeff_lengths(self.signal_length, self.wavelet_name, 0, mode='symmetric')
        self.assertGreater(len_cA, 0)
        self.assertGreater(len_cD, 0)

        # Test invalid level
        with self.assertRaises(ValueError):
            get_wavedec_coeff_lengths(self.signal_length, self.wavelet_name, -1, mode='symmetric')

        # Test invalid wavelet
        with patch('pywt.Wavelet') as mock_wavelet:
            mock_wavelet.side_effect = ValueError("Invalid wavelet")
            with self.assertRaises(ValueError):
                get_wavedec_coeff_lengths(self.signal_length, 'invalid_wavelet', self.dwt_level)

    def test_time_series_embedding(self):
        """Test TimeSeriesEmbedding layer."""
        embedding_layer = TimeSeriesEmbedding(maxlen=self.maxlen, embed_dim=self.embed_dim)
        input_tensor = tf.zeros((self.batch_size, self.maxlen, 1))
        output = embedding_layer(input_tensor)
        
        # Check output shape
        expected_shape = (self.batch_size, self.maxlen, self.embed_dim)
        self.assertEqual(output.shape, expected_shape, f"Expected shape {expected_shape}, got {output.shape}")

        # Check positional encoding
        pos_encoding = embedding_layer.get_sinusoidal_pos_encoding(self.maxlen, self.embed_dim)
        self.assertEqual(pos_encoding.shape, (1, self.maxlen, self.embed_dim))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(pos_encoding)))

        # Test invalid input shape
        with self.assertRaises(ValueError):
            invalid_input = tf.zeros((self.batch_size, self.maxlen + 1, 1))
            embedding_layer(invalid_input)

        # Test 2D input
        input_2d = tf.zeros((self.batch_size, self.maxlen))
        output_2d = embedding_layer(input_2d)
        self.assertEqual(output_2d.shape, expected_shape)

    def test_downsample_transformer_block(self):
        """Test DownsampleTransformerBlock layer."""
        transformer_block = DownsampleTransformerBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            retention_rate=self.retention_rate,
            rate=0.3
        )
        input_tensor = tf.random.normal((self.batch_size, self.maxlen, self.embed_dim))
        output, indices = transformer_block(input_tensor, training=False)

        # Check output shape
        expected_seq_len = max(1, int(round(self.maxlen * self.retention_rate)))
        expected_shape = (self.batch_size, expected_seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape, f"Expected shape {expected_shape}, got {output.shape}")
        self.assertEqual(indices.shape, (self.batch_size, expected_seq_len))

        # Check indices are valid
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < self.maxlen))

        # Test invalid retention rate
        with self.assertRaises(ValueError):
            DownsampleTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                retention_rate=0.0
            )

        # Test invalid input shape
        with self.assertRaises(ValueError):
            invalid_input = tf.zeros((self.batch_size, self.maxlen))
            transformer_block(invalid_input)

    def test_wavelet_downsampling_model(self):
        """Test WaveletDownsamplingModel."""
        len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(self.original_length, self.wavelet_name, self.dwt_level, 'symmetric')
        detail_transformer = build_detail_transformer(
            signal_coeffs_len,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=1,
            retention_rate=self.retention_rate
        )
        model = WaveletDownsamplingModel(
            detail_transformer_model=detail_transformer,
            wavelet_name=self.wavelet_name,
            approx_ds_factor=2,
            original_length=self.original_length,
            signal_coeffs_len=signal_coeffs_len,
            dwt_level=self.dwt_level,
            normalize_details=True,
            decomposition_mode='symmetric'
        )
        
        # Build model
        input_shape = (None, self.original_length, 1)
        model.build(input_shape)
        
        # Test forward pass
        input_tensor = tf.random.normal((self.batch_size, self.original_length, 1))
        with patch.object(model, '_decompose_batch_py_func') as mock_decompose:
            mock_decompose.return_value = (
                np.random.normal(size=(self.batch_size, len_cA)).astype(np.float32),
                np.random.normal(size=(self.batch_size, signal_coeffs_len)).astype(np.float32)
            )
            output = model(input_tensor, training=False, return_indices=False)
        
        # Check output shape
        expected_output_len = model.combined_ds_len
        self.assertEqual(output.shape, (self.batch_size, expected_output_len))

        # Test with indices
        output, indices = model(input_tensor, training=False, return_indices=True)
        self.assertIsInstance(indices, list)
        self.assertGreater(len(indices), 0)

        # Test invalid input shape
        with self.assertRaises(ValueError):
            invalid_input = tf.zeros((self.batch_size, self.original_length))
            model(invalid_input)

        # Test normalization
        model_no_norm = WaveletDownsamplingModel(
            detail_transformer_model=detail_transformer,
            wavelet_name=self.wavelet_name,
            approx_ds_factor=2,
            original_length=self.original_length,
            signal_coeffs_len=signal_coeffs_len,
            dwt_level=self.dwt_level,
            normalize_details=False,
            decomposition_mode='symmetric'
        )
        model_no_norm.build(input_shape)
        output_no_norm = model_no_norm(input_tensor, training=False, return_indices=False)
        self.assertEqual(output_no_norm.shape, (self.batch_size, expected_output_len))

    def test_downsampling_loss(self):
        """Test downsampling_loss function."""
        y_true = tf.random.normal((self.batch_size, self.maxlen))
        y_pred = tf.random.normal((self.batch_size, self.maxlen))
        loss = downsampling_loss(y_true, y_pred)
        
        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreater(loss, 0.0)
        self.assertFalse(tf.math.is_nan(loss))

        # Test with identical inputs
        loss_zero = downsampling_loss(y_true, y_true)
        self.assertAlmostEqual(float(loss_zero), 0.0, places=5)

    @patch('streaming.json')
    @patch('streaming.logger')
    @patch('streaming.psutil')
    @patch('streaming.tf')
    @patch('streaming.keras')
    @patch('streaming.np')
    

    def test_deserialization_schema(self):
        """Test DeserializationSchema class."""
        deserializer = DeserializationSchema()
        valid_message = json.dumps([1.0, 2.0, 3.0]).encode('utf-8')
        result = deserializer.deserialize(valid_message)
        self.assertEqual(result, [1.0, 2.0, 3.0])

        # Test None message
        result = deserializer.deserialize(None)
        self.assertEqual(result, [])

        # Test invalid JSON
        invalid_message = b'invalid'
        with patch('streaming.logger') as mock_logger:
            result = deserializer.deserialize(invalid_message)
            self.assertEqual(result, [])
            mock_logger.error.assert_called()

    def test_serialization_schema(self):
        """Test SerializationSchema class."""
        serializer = SerializationSchema(topic='test-topic')
        valid_element = [1.0, 2.0, 3.0]
        result = serializer.serialize(valid_element)
        self.assertEqual(result, json.dumps(valid_element).encode('utf-8'))

        # Test empty element
        result = serializer.serialize([])
        self.assertEqual(result, b'')

        # Test None element
        result = serializer.serialize(None)
        self.assertEqual(result, b'')

        # Test non-finite values
        element_nan = [float('nan'), 1.0, float('inf')]
        result = serializer.serialize(element_nan)
        self.assertEqual(result, json.dumps([0.0, 1.0, 0.0]).encode('utf-8'))

if __name__ == '__main__':
    unittest.main()