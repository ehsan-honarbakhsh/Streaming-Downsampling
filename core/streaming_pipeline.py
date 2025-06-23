import json
import numpy as np
import yaml
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink, KafkaRecordSerializationSchema
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.common import WatermarkStrategy, Configuration
from pyflink.common.serialization import DeserializationSchema, SerializationSchema
from pyflink.java_gateway import get_gateway
from pyflink.datastream.connectors.kafka import DeliveryGuarantee
from .downsampling_algorithm2 import WaveletDownsamplingModel, TimeSeriesEmbedding, DownsampleTransformerBlock, get_wavedec_coeff_lengths, downsampling_loss
import tensorflow as tf
import keras
import os
import traceback
import logging
import psutil
import time
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streaming_pipeline.log', mode='w', delay=True)
    ]
)
logger = logging.getLogger(__name__)

class DeserializationSchema(DeserializationSchema):
    """Custom deserialization schema for Kafka messages."""
    def __init__(self):
        super().__init__(Types.STRING())
        gateway = get_gateway()
        self._j_deserialization_schema = gateway.jvm.org.apache.flink.api.common.serialization.SimpleStringSchema()

    def deserialize(self, message: bytes) -> list:
        if message is None:
            logger.warning("Received None message in DeserializationSchema")
            return []
        try:
            result = json.loads(message.decode('utf-8'))
            return result
        except Exception as e:
            logger.error(f"Error deserializing message: {message}. Error: {e}")
            return []

    def open(self, context):
        pass

    def get_produced_type(self):
        return Types.LIST(Types.FLOAT())

class SerializationSchema(SerializationSchema):
    """Custom serialization schema for Kafka messages."""
    def __init__(self, topic='m4-downsampled-topic'):
        super().__init__(Types.STRING())
        self.topic = topic
        gateway = get_gateway()
        j_string_serialization_schema = gateway.jvm.org.apache.flink.api.common.serialization.SimpleStringSchema()
        self._j_serialization_schema = gateway.jvm.org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema.builder() \
            .setTopic(self.topic) \
            .setValueSerializationSchema(j_string_serialization_schema) \
            .build()

    def serialize(self, element: list) -> bytes:
        if element is None or len(element) == 0:
            logger.warning("Serializing empty or None element")
            return b''
        try:
            safe_element = [0.0 if not np.isfinite(x) else x for x in element]
            json_string = json.dumps(safe_element)
            return json_string.encode('utf-8')
        except Exception as e:
            logger.error(f"Error serializing element: {element}. Error: {e}")
            return b''

    def open(self, context):
        pass

    def get_produced_type(self):
        return Types.STRING()

def load_kafka_config(config_path='kafka_config.yaml'):
    """Load Kafka and Flink configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration with defaults if file is missing or invalid.
    """
    default_config = {
        'bootstrap_servers': 'localhost:9092',
        'num_partitions': 2,
        'replication_factor': 1,
        'batch_size': 32768,
        'linger_ms': 2,
        'compression_type': 'gzip',
        'buffer_memory': 33554432,
        'chunk_size': 20,  # Default chunk size for processing downsampled outputs
        'chunk_size_fraction': 0.1,  # Target fraction of output size for dynamic chunking
        'min_chunk_size': 10,  # Minimum chunk size to prevent too small chunks
        'max_chunk_size': 100  # Maximum chunk size to prevent memory issues
    }
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            logger.warning(f"Empty config file at {config_path}, using defaults")
            return default_config
        logger.info(f"Loaded config from {config_path}: {config}")
        for key, value in default_config.items():
            config.setdefault(key, value)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return default_config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return default_config

def setup_streaming_pipeline(X_train_normalized, X_test_normalized, model, original_length=150, input_topic='m4-input-topic', output_topic='m4-downsampled-topic', config_path='kafka_config.yaml'):
    """Set up a streaming pipeline with Kafka and Flink for time series downsampling.

    Args:
        X_train_normalized (np.ndarray): Normalized training data.
        X_test_normalized (np.ndarray): Normalized test data.
        model (keras.Model): Downsampling model.
        original_length (int): Expected length of input series.
        input_topic (str): Kafka input topic name.
        output_topic (str): Kafka output topic name.
        config_path (str): Path to configuration YAML file.
    """
    logger.info(f"Validating input data: X_train_normalized shape={X_train_normalized.shape}, X_test_normalized shape={X_test_normalized.shape}")
    if np.any(np.isnan(X_train_normalized)) or np.any(np.isinf(X_train_normalized)):
        logger.warning("NaN or Inf detected in X_train_normalized, replacing with 0")
        X_train_normalized = np.where(np.isnan(X_train_normalized) | np.isinf(X_train_normalized), 0, X_train_normalized)
    if np.any(np.isnan(X_test_normalized)) or np.any(np.isinf(X_test_normalized)):
        logger.warning("NaN or Inf detected in X_test_normalized, replacing with 0")
        X_test_normalized = np.where(np.isnan(X_test_normalized) | np.isinf(X_test_normalized), 0, X_test_normalized)
    logger.info(f"Sample X_train_normalized[0]: {X_train_normalized[0][:10]}")
    logger.info(f"Sample X_test_normalized[0]: {X_test_normalized[0][:10]}")

    config = load_kafka_config(config_path)
    bootstrap_servers = config['bootstrap_servers']
    num_partitions = config['num_partitions']
    replication_factor = config['replication_factor']
    parallelism = max(1, psutil.cpu_count())
    chunk_size = config['chunk_size']
    chunk_size_fraction = config['chunk_size_fraction']
    min_chunk_size = config['min_chunk_size']
    max_chunk_size = config['max_chunk_size']
    logger.info(f"Using parallelism: {parallelism}, chunk_size: {chunk_size}, chunk_size_fraction: {chunk_size_fraction}")

    admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    topic_list = [
        NewTopic(name=input_topic, num_partitions=num_partitions, replication_factor=replication_factor),
        NewTopic(name=output_topic, num_partitions=num_partitions, replication_factor=replication_factor)
    ]
    try:
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        logger.info(f"Created Kafka topics {input_topic} and {output_topic} with {num_partitions} partitions")
    except Exception as e:
        logger.info(f"Topics {input_topic} and/or {output_topic} already exist or error: {e}")
    finally:
        admin_client.close()

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, allow_nan=True).encode('utf-8'),
        key_serializer=lambda k: str(k).encode('utf-8'),
        batch_size=config['batch_size'],
        linger_ms=config['linger_ms'],
        compression_type=config['compression_type'],
        buffer_memory=config['buffer_memory']
    )

    start_time = time.time()
    for idx, series in enumerate(np.concatenate([X_train_normalized, X_test_normalized], axis=0)):
        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
            logger.warning(f"Series {idx} contains NaN/Inf, replacing with 0")
            series = np.where(np.isnan(series) | np.isinf(series), 0, series)
        producer.send(input_topic, key=str(idx), value=series.tolist())
        if idx % 5 == 0:
            logger.info(f"Sent {idx} series to Kafka")
    producer.flush(timeout=30.0)
    producer.close()
    logger.info(f"Finished streaming {len(X_train_normalized) + len(X_test_normalized)} series to Kafka in {time.time() - start_time:.2f} seconds")

    model_path = os.path.join(".", "downsampling_model.keras")
    try:
        model.save(model_path)
        logger.info(f"Saved model to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

    config = Configuration()
    config.set_string(
        "pipeline.jars",
        "file:///Users/ehsanhonarbakhsh/Documents/GitHub/Downsampling/flink-connector-kafka-4.0.0-2.0.jar;"
        "file:///Users/ehsanhonarbakhsh/Documents/GitHub/Downsampling/kafka-clients-4.0.0.jar"
    )
    config.set_string("log4j.logger.org.apache.flink", "INFO")
    config.set_integer("taskmanager.numberOfTaskSlots", parallelism)
    logger.info(f"Setting Flink taskmanager.numberOfTaskSlots to {parallelism}")

    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_parallelism(parallelism)
    env.get_config().set_auto_watermark_interval(1000)

    deserializer = DeserializationSchema()
    kafka_source = KafkaSource.builder() \
        .set_bootstrap_servers(bootstrap_servers) \
        .set_topics(input_topic) \
        .set_group_id('m4-flink-group') \
        .set_property("auto.offset.reset", "earliest") \
        .set_property("enable.auto.commit", "true") \
        .set_property("auto.commit.interval.ms", "1000") \
        .set_value_only_deserializer(deserializer) \
        .build()

    data_stream = env.from_source(kafka_source, WatermarkStrategy.no_watermarks(), "Kafka Source")

    def _get_model():
        if not hasattr(_get_model, 'model'):
            logger.info("Loading model in _get_model")
            mem_usage = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage before model load: {mem_usage:.2f} MB")
            try:
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
                _get_model.model = keras.models.load_model(
                    model_path,
                    custom_objects={
                        "WaveletDownsamplingModel": WaveletDownsamplingModel,
                        "TimeSeriesEmbedding": TimeSeriesEmbedding,
                        "DownsampleTransformerBlock": DownsampleTransformerBlock,
                        "downsampling_loss": downsampling_loss
                    }
                )
                logger.info("Model loaded successfully")
                mem_usage = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage after model load: {mem_usage:.2f} MB")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.error(traceback.format_exc())
                raise
            finally:
                logger.info("Flushing logs after model load")
                for handler in logger.handlers:
                    handler.flush()
        return _get_model.model

    def process_element(element):
        """Process a single time series element through the downsampling model."""
        if not hasattr(process_element, 'count'):
            process_element.count = 0
        process_element.count += 1
        logger.info(f"Processing element {process_element.count}: Starting")

        try:
            process = psutil.Process()
            mem_usage = process.memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            logger.info(f"Resource usage: Memory={mem_usage:.2f}MB, CPU={cpu_percent:.1f}%")

            if isinstance(element, str):
                element = json.loads(element)
            if not isinstance(element, list):
                logger.error(f"Invalid input: Expected list, got {type(element)}")
                return json.dumps([])

            element = [0.0 if x is None or not np.isfinite(x) else float(x) for x in element]
            if not all(isinstance(x, (int, float)) for x in element):
                logger.error(f"Element contains non-numeric values after cleaning: {element[:10]}")
                return json.dumps([])

            series = np.array(element, dtype=np.float32)
            if series.shape[0] != original_length:
                logger.warning(f"Reshaping series from {series.shape[0]} to {original_length}")
                if series.shape[0] > original_length:
                    series = series[:original_length]
                else:
                    series = np.pad(series, (0, original_length - series.shape[0]), mode='constant', constant_values=0)
            model_input = series.reshape(1, original_length, 1)

            model = _get_model()
            if model is None:
                logger.error("Model is None")
                return json.dumps([])

            logger.info("Calling model inference...")
            with tf.device('/CPU:0'):
                downsampled = model.call(tf.convert_to_tensor(model_input), training=False, return_indices=False)
            logger.info(f"Downsampled output shape: {downsampled.shape}, first 10 values: {downsampled.numpy().flatten()[:10]}")

            if tf.reduce_any(tf.math.is_nan(downsampled)) or tf.reduce_any(tf.math.is_inf(downsampled)):
                logger.error(f"NaN or Inf detected in downsampled output: {downsampled.numpy().flatten()[:10]}")
                return json.dumps([])

            # Determine chunk size dynamically based on output size
            output_size = downsampled.shape[1]
            #dynamic_chunk_size = int(max(min_chunk_size, min(max_chunk_size, output_size * chunk_size_fraction)))
            dynamic_chunk_size = chunk_size
            #effective_chunk_size = chunk_size if chunk_size > 0 else dynamic_chunk_size
            effective_chunk_size= dynamic_chunk_size
            logger.info(f"Using chunk_size={effective_chunk_size} for output_size={output_size}")

            result = []
            for i in range(0, downsampled.shape[1], effective_chunk_size):
                chunk = downsampled[:, i:i+effective_chunk_size].numpy().flatten().tolist()
                result.extend(chunk)
                del chunk
                gc.collect()

            if not result:
                logger.error("Result is empty")
                return json.dumps([])
            logger.info(f"Downsampled output length: {len(result)}")
            logger.info(f"Serializing result: first 5 values={result[:5]}, length={len(result)}")
            serialized = json.dumps(result, allow_nan=False)
            logger.info(f"Completed processing element {process_element.count}")

            del downsampled, model_input, series
            gc.collect()
            keras.backend.clear_session()

            for handler in logger.handlers:
                handler.flush()

            return serialized
        except Exception as e:
            logger.error(f"Error in element {process_element.count}: {e}")
            logger.error(traceback.format_exc())
            for handler in logger.handlers:
                handler.flush()
            return json.dumps([])

    logger.info("Mapping process_element to data stream")
    try:
        processed_stream = data_stream.map(process_element, output_type=Types.STRING())
    except Exception as e:
        logger.error(f"Error mapping process_element: {e}")
        logger.error(traceback.format_exc())
        raise

    serializer = SerializationSchema(topic=output_topic)
    kafka_sink = KafkaSink.builder() \
        .set_bootstrap_servers(bootstrap_servers) \
        .set_record_serializer(serializer) \
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE) \
        .build()

    processed_stream.sink_to(kafka_sink)

    logger.info(f"Executing Flink pipeline to process {len(X_train_normalized) + len(X_test_normalized)} elements with parallelism {parallelism}")
    try:
        env.execute("Streaming Downsampling Pipeline")
        logger.info(f"Processed approximately {len(X_train_normalized) + len(X_test_normalized)} elements")
    except Exception as e:
        logger.error(f"Flink pipeline execution failed: {e}")
        logger.error(traceback.format_exc())
        raise