import csv
import json
import time
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_producer(bootstrap_servers='localhost:9092'):
    """Create and return a Kafka producer with error handling."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=5,  # Retry on failure
            acks='all'  # Ensure all replicas acknowledge
        )
        logger.info("Kafka producer created successfully")
        return producer
    except KafkaError as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise

def send_csv_to_kafka(csv_file, topic, delay=0.1):
    """Read CSV file and send rows to Kafka topic with specified delay."""
    producer = create_producer()
    try:
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                try:
                    producer.send(topic, value=row)
                    logger.info(f"Sent: {row}")
                    time.sleep(delay)  # Simulate real-time streaming
                except KafkaError as e:
                    logger.error(f"Failed to send message: {e}")
        producer.flush()
        logger.info("All messages flushed to Kafka")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        producer.close()
        logger.info("Kafka producer closed")

if __name__ == "__main__":
    # Configuration
    CSV_FILE = 'M4/Daily/Daily-train.csv'
    KAFKA_TOPIC = 'csv-data'
    DELAY = 0.1  # Delay in seconds between messages

    try:
        send_csv_to_kafka(CSV_FILE, KAFKA_TOPIC, DELAY)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")