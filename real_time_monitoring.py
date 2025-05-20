import json
import numpy as np
from kafka import KafkaConsumer
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go
from collections import deque
import threading
import time

# Configuration
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
OUTPUT_TOPIC = 'm4-downsampled-topic'
CONSUMER_GROUP = 'visualization-group'
EXPECTED_DOWNsampled_LENGTH = 43  # Adjust based on model output
MAX_SERIES = 10
POLL_TIMEOUT_MS = 100
UPDATE_INTERVAL_MS = 100

# Buffer for downsampled series
recent_series = deque(maxlen=MAX_SERIES)

def safe_deserialize_value(x):
    if x is None:
        print("Received None value in value_deserializer")
        return []
    try:
        return json.loads(x.decode('utf-8'))
    except Exception as e:
        print(f"Value deserialization error: {e}, raw value: {x}")
        return []

def safe_deserialize_key(x):
    if x is None:
        print("Received None key in key_deserializer")
        return ""
    try:
        return x.decode('utf-8')
    except Exception as e:
        print(f"Key deserialization error: {e}, raw key: {x}")
        return ""

def consume_kafka():
    try:
        consumer = KafkaConsumer(
            OUTPUT_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=CONSUMER_GROUP,
            auto_offset_reset='earliest',
            value_deserializer=safe_deserialize_value,
            key_deserializer=safe_deserialize_key
        )
        print(f"Started Kafka consumer for {OUTPUT_TOPIC}")
        message_count = {'total': 0, 'valid': 0, 'skipped': 0}

        while True:
            messages = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)
            for topic_partition, partition_messages in messages.items():
                print(f"Received {len(partition_messages)} messages from partition {topic_partition}")
                for message in partition_messages:
                    key = message.key
                    message_count['total'] += 1
                    try:
                        series = np.array(message.value, dtype=np.float32)
                        if len(series) == 0:
                            print(f"Empty series for key {key}, skipping")
                            message_count['skipped'] += 1
                            continue
                        message_count['valid'] += 1
                        print(f"Received downsampled series {key}, length={len(series)}, total messages={message_count['total']}, valid={message_count['valid']}")
                        recent_series.append(series)
                    except Exception as e:
                        print(f"Error processing message {key}: {e}, raw value={message.value}")
                        message_count['skipped'] += 1
            print(f"Recent series buffer size: {len(recent_series)}, message counts: {message_count}")
            time.sleep(0.1)

    except Exception as e:
        print(f"Kafka consumer error: {e}")
        time.sleep(1)

def run_visualization():
    # Start Kafka consumer thread
    print("Starting Kafka consumer thread...")
    threading.Thread(target=consume_kafka, daemon=True).start()

    # Initialize Dash app
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("Real-Time Downsampled Time Series Visualization"),
        dcc.Graph(id='downsampled-plot'),
        dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL_MS, n_intervals=0)
    ])

    @app.callback(
        Output('downsampled-plot', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_plot(n):
        print(f"Updating plot, n_intervals={n}, recent_series size={len(recent_series)}")
        if not recent_series:
            print("No data in recent_series, returning empty figure")
            return go.Figure()

        fig = go.Figure()
        for i, downsampled in enumerate(recent_series):
            if not isinstance(downsampled, np.ndarray):
                print(f"Invalid series type at index {i}: {type(downsampled)}")
                continue
            print(f"Plotting downsampled series {i+1}: length={len(downsampled)}, values={downsampled[:5]}")
            fig.add_trace(go.Scatter(
                x=list(range(len(downsampled))),
                y=downsampled,
                name=f'Downsampled {i+1}',
                mode='lines+markers',
                marker=dict(size=8),
                line=dict(color='blue', dash='dot'),
                opacity=0.7
            ))

        fig.update_layout(
            title='Real-Time Downsampled Time Series',
            xaxis_title='Sample',
            yaxis_title='Amplitude',
            showlegend=True,
            height=600
        )
        print("Plot updated successfully")
        return fig

    print("Starting Dash server on http://localhost:8050...")
    try:
        app.run(debug=False, host='0.0.0.0', port=8050)
    except Exception as e:
        print(f"Failed to start Dash server: {e}")

if __name__ == "__main__":
    dash_thread = threading.Thread(target=run_visualization, daemon=True)
    dash_thread.start()
    print("Dash dashboard running at http://localhost:8050")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down Dash server.")