import json
import numpy as np
from kafka import KafkaProducer
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink, KafkaRecordSerializationSchema
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.common import WatermarkStrategy, Configuration
from pyflink.common.serialization import DeserializationSchema, SerializationSchema
from pyflink.java_gateway import get_gateway
from pyflink.datastream.connectors.kafka import DeliveryGuarantee
from .downsampling_algorithm import WaveletDownsamplingModel, TimeSeriesEmbedding, DownsampleTransformerBlock, get_wavedec_coeff_lengths
import tensorflow as tf
import keras
import os

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

def setup_streaming_pipeline(X_train_normalized, X_test_normalized, model, original_length=150, input_topic='m4-input-topic', output_topic='m4-downsampled-topic'):
    # Kafka Producer: Stream M4 dataset to Kafka topic
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: str(k).encode('utf-8'),
        batch_size=16384,
        linger_ms=5,
        compression_type='gzip'
    )
    
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

    # Save the model for Flink processing
    model_path = os.path.join(".", "downsampling_model.keras")
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