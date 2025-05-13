# Streaming Downsampling
Architecture Overview:
The core concept behind this algorithm was inspired by the image processing coursework I completed in Semester 2, where I developed an image denoising algorithm using detail coefficients (a project that received positive feedback from my lecturer).
Building on that experience, and after reviewing several relevant research papers and methodologies, I propose a similar approach for time series data. The signal is first decomposed using wavelet transforms into approximation and detail coefficients. Previous studies have typically relied on approximation coefficients for downsampling, as they effectively capture the general trend of the signal. However, this often leads to the loss of critical information such as peaks and dips,features that are essential for accurate representation and analysis.
To overcome this limitation, I argue that detail coefficients must be included in the downsampling process. Therefore, my approach focuses on processing the detail coefficients using a Transformer-based model, enabling downsampling in the frequency domain while preserving fine-grained signal features.
The choice of a Transformer-based model is deliberate. Transformers are well-suited for learning complex patterns in sequential data and are capable of capturing long-range dependencies between data points,an essential characteristic when dealing with time series. Moreover, their adaptive nature allows them to dynamically recognize and retain significant features within varying datasets. The transformer can potentially learn which high-frequency details are most critical to retain, tailoring the downsampling to the data’s specific characteristics.
This architecture combines the multi-resolution capability of wavelet transforms with the powerful pattern recognition and sequence modeling abilities of Transformers. Compared to traditional methods, this approach aims to achieve a balanced representation of both the signal’s overall trend and its high-frequency components.This forms the core of my downsampling architecture.
To make the architecture suitable for stream processing, I incorporated Apache Kafka and Apache Flink, which together provide a robust infrastructure for handling real-time data streams. For this initial prototype, I set up two Kafka topics: one for input data and another for storing the processed output (sink). Apache Flink acts as the processing engine,reading data from the Kafka source topic, passing it through the proposed architecture for downsampling, and writing the results to the output topic.

How the Algorithm Works:
Wavelet Transform:
 It starts by applying a wavelet transform to decompose the time series into two types of coefficients:
Approximation Coefficients: These represent the low-frequency components, capturing the smooth, overall trend of the data.
Detail Coefficients: These capture the high-frequency components, such as noise, sudden changes, or fine details.
Retaining Approximation Coefficients:
I keep the approximation coefficients as they are, preserving the general behavior of the time series. Since these coefficients are fewer in number (depending on the decomposition level), this step naturally reduces the data size to some extent.
Processing Detail Coefficients with a Transformer:
 The detail coefficients are passed to a transformer model, which downsamples them. Transformers are adept at capturing complex patterns and dependencies, so this step likely compresses or selects the most important high-frequency details into a lower-dimensional representation.
Reconstruction:
 Finally, I combine the retained approximation coefficients with the downsampled detail coefficients to produce a reduced version of the original time series.

