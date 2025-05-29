# Streaming Downsampling
Industrial data centers are the nerve centers of modern industry, housing critical IT infrastructure that supports everything from manufacturing processes and supply chain management to energy grids and smart city initiatives. These facilities generate an ever-increasing deluge of time series data from a multitude of sensors and operational systems . This data, tracking metrics like server temperatures, power consumption, network traffic, equipment status, and environmental conditions, arrives at high frequencies, often multiple readings per second. While this granular data is invaluable for real-time operational control and immediate anomaly detection, managing, storing, and analyzing such vast quantities presents significant challenges .This is where downsampling emerges not just as a useful technique, but as a critical necessity.

# The project can be divided into three main components:
## Downsampling Model: 
This is the core of the project and consists of the primary architecture for the downsampling algorithm.


## Streaming Model:
This component implements a streaming pipeline using Apache Kafka and Apache Flink to enhance parallelism, thereby improving the overall performance and execution speed of the system.


## Real-time Visualization Model: 
This module provides a real-time visual representation of the output, specifically the downsampled data.


## 1.Downsampling Model

The most critical component of this project is the downsampling algorithm. The downsampling model is constructed by integrating two primary components: a Wavelet Transform and a Transformer.

### Block 1: Wavelet Transform
The downsampling model begins by applying a Wavelet Transform to decompose the time series into two distinct sets of coefficients:
Approximation Coefficients: Representing the low-frequency components, these coefficients capture the smooth, overall trend of the data .
Detail Coefficients: These encapsulate the high-frequency components, such as noise, abrupt changes, or fine details.
 
The approximation coefficients are retained in their original form to preserve the general structure and behavior of the time series.

### Block 2: Transformer
The detail coefficients are then passed through a Transformer-based model comprising two attention layers, a feed-forward neural network, and an importance scoring mechanism. This importance scorer assigns a relevance value to each data point within the sequence. The Transformer architecture, introduced by Vaswani et al. (2017), leverages self-attention mechanisms to model complex dependencies, making it well-suited for capturing intricate patterns in high-frequency components. Subsequently, downsampling is achieved by applying a predefined threshold to retain only the most important and representative data points. This step effectively condenses significant high-frequency information into a lower-dimensional representation. Finally, the retained approximation coefficients are combined with the downsampled detail coefficients to reconstruct a reduced version of the original time series.

## Strengths of the Downsampling Algorithm

### Multi-Scale Feature Preservation:

By employing wavelet decomposition, the proposed method explicitly separates trend components from detailed variations in the signal. This enables the algorithm to consider both coarse and fine-grained patterns during downsampling, offering a significant advantage over conventional approaches that treat all data uniformly.

### Adaptive Handling of High-Frequency Information:

Integrating a Transformer for processing detail coefficients introduces adaptability into the model. The Transformer’s attention mechanism can learn to prioritize which high-frequency elements are most critical to retain, making the downsampling process more context-aware and data-specific.

### High-Fidelity Signal Retention:

Provided the Transformer is effectively designed and trained, the algorithm has the potential to preserve essential features,both global trends and select high-frequency details,that may be lost in simpler downsampling techniques. This makes the approach particularly suitable for applications where maintaining signal integrity is paramount.
	

## 2.Streaming Model
The objective of designing the streaming pipeline is to enhance the performance of the downsampling model by enabling near real-time processing through a fully parallelized architecture . The streaming workflow begins with Apache Kafka, which ingests time series data from a designated data source and publishes it to Kafka topics. Apache Flink then consumes the data from these Kafka topics, applies the downsampling model to the continuous data stream, and publishes the processed (downsampled) output to a Kafka sink. The sink can be configured as either a persistent storage system or another Kafka topic for further downstream processing. The inherent parallelism offered by Apache Kafka, Apache Flink, and the Transformer-based downsampling model significantly accelerates the end-to-end pipeline. This includes data ingestion, downsampling, storage, and ultimately, real-time monitoring and visualization. By leveraging distributed computing at each stage, the system achieves high throughput and low latency, making it well-suited for time-sensitive applications .
The built-in application state management in Apache Flink ensures that no data is lost at any stage, as Flink periodically writes consistent checkpoints of the application state to a remote and durable storage system

## 3.Real-Time Visualization
To facilitate the monitoring of the downsampled data stream, a visualization component has been developed. This component includes an algorithm that consumes the downsampled data from the Kafka topic and visualizes it in near real-time using the Dash library in Python . Dash is an open-source framework designed for building interactive, web-based data visualization applications. By integrating Dash with the Kafka consumer, the system enables dynamic and continuous updates of the visual output as new data points are produced by the downsampling pipeline. This setup provides an intuitive and responsive interface for observing the behavior of the downsampling algorithm and verifying its effectiveness in real time



