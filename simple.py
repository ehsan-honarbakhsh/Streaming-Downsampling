import numpy as np
import json
import time
import threading
from kafka import KafkaProducer, KafkaConsumer
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go
from collections import deque

# Configuration
INPUT_TOPIC = 'input-topic'
OUTPUT_TOPIC = 'output-topic'
BOOTSTRAP_SERVERS = 'localhost:9092'
NUM_SERIES = 20  # Number of time series to generate
SERIES_LENGTH = 100  # Points per series
UPDATE_INTERVAL = 1000  # Dashboard update interval (ms)

# Generate synthetic time series data (sine wave + noise)
def generate_time_series(num_series, length):
    x = np.linspace(0, 4 * np.pi, length)
    series = [np.sin(x + np.random.uniform(0, 2 * np.pi)) + np.random.normal(0, 0.1, length)
              for _ in range(num_series)]
    return np.array(series)

# Stream series to Kafka input topic
def stream_to_kafka():
    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8'),
            batch_size=16384,
            linger_ms=5
        )
        series_data = generate_time_series(NUM_SERIES, SERIES_LENGTH)
        print(f"Streaming {NUM_SERIES} time series to {INPUT_TOPIC}")
        for idx, series in enumerate(series_data):
            producer.send(INPUT_TOPIC, key=str(idx), value=series.tolist())
            print(f"Sent series {idx} to {INPUT_TOPIC}")
            time.sleep(0.1)  # Simulate real-time streaming
        producer.flush()
        producer.close()
        print("Finished streaming to Kafka")
    except Exception as e:
        print(f"Kafka producer error: {e}")

# Process series (consume from input, scale, produce to output)
def process_kafka():
    try:
        consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id='processing-group',
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8')
        )
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8')
        )
        print(f"Processing series from {INPUT_TOPIC} to {OUTPUT_TOPIC}")
        for message in consumer:
            key = message.key
            original = np.array(message.value)
            processed = original * 0.5  # Simple transformation (scale by 0.5)
            producer.send(OUTPUT_TOPIC, key=key, value=processed.tolist())
            print(f"Processed series {key}: original mean={original.mean():.2f}, processed mean={processed.mean():.2f}")
            time.sleep(0.1)
        producer.flush()
        producer.close()
    except Exception as e:
        print(f"Kafka processing error: {e}")

# Consume input and output topics for visualization
input_buffer = {}
output_buffer = {}
recent_series = deque(maxlen=10)  # Store up to 5 recent series

def consume_kafka():
    try:
        input_consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id='visualization-group',
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8')
        )
        output_consumer = KafkaConsumer(
            OUTPUT_TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id='visualization-group',
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8')
        )
        print(f"Consuming {INPUT_TOPIC} and {OUTPUT_TOPIC} for visualization")
        while True:
            # Poll input topic
            input_messages = input_consumer.poll(timeout_ms=1000)
            if not input_messages:
                print("No new messages from input-topic")
            for topic_partition, messages in input_messages.items():
                for message in messages:
                    key = message.key
                    input_buffer[key] = np.array(message.value)
                    print(f"Received input series {key}, length={len(input_buffer[key])}")
                    if key in output_buffer:
                        recent_series.append((input_buffer[key], output_buffer[key]))
                        print(f"Paired series {key} for visualization")
                        del input_buffer[key]
                        del output_buffer[key]
            # Poll output topic
            output_messages = output_consumer.poll(timeout_ms=1000)
            if not output_messages:
                print("No new messages from output-topic")
            for topic_partition, messages in output_messages.items():
                for message in messages:
                    key = message.key
                    output_buffer[key] = np.array(message.value)
                    print(f"Received output series {key}, length={len(output_buffer[key])}")
                    if key in input_buffer:
                        recent_series.append((input_buffer[key], output_buffer[key]))
                        print(f"Paired series {key} for visualization")
                        del input_buffer[key]
                        del output_buffer[key]
            print(f"Current recent_series size: {len(recent_series)}")
            time.sleep(0.1)
    except Exception as e:
        print(f"Kafka consumer error: {e}")
        time.sleep(1)

# Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Real-Time Time Series Visualization"),
    dcc.Graph(id='time-series-plot'),
    dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL, n_intervals=0)
])

@app.callback(
    Output('time-series-plot', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_plot(n):
    print(f"Updating plot, n_intervals={n}, recent_series size={len(recent_series)}")
    if not recent_series:
        print("No data in recent_series, returning empty figure")
        return go.Figure()
    fig = go.Figure()
    for i, (original, processed) in enumerate(recent_series):
        print(f"Adding series {i+1}: original length={len(original)}, processed length={len(processed)}")
        fig.add_trace(go.Scatter(
            x=list(range(len(original))),
            y=original,
            name=f'Original {i+1}',
            line=dict(color='orange', dash='solid'),
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(processed))),
            y=processed,
            name=f'Processed {i+1}',
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(color='blue', dash='dot'),
            opacity=0.7
        ))
    fig.update_layout(
        title='Original vs Processed Time Series',
        xaxis_title='Sample',
        yaxis_title='Amplitude',
        showlegend=True,
        height=600
    )
    print("Plot updated successfully")
    return fig

# Run Dash server
def run_dash():
    print("Starting Dash server...")
    try:
        app.run(debug=False, host='0.0.0.0', port=8050)
        print("Dash server running at http://localhost:8050")
    except Exception as e:
        print(f"Dash server error: {e}")

if __name__ == "__main__":
    # Start threads
    threading.Thread(target=stream_to_kafka, daemon=True).start()
    threading.Thread(target=process_kafka, daemon=True).start()
    threading.Thread(target=consume_kafka, daemon=True).start()
    threading.Thread(target=run_dash, daemon=True).start()
    
    # Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
