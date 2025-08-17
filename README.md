# Normalizing Flows for Regular Time Series Forecasting

[cite_start]This repository contains the official source code, experiments, and final paper for the thesis titled **"Normalizing Flows for Regular Time Series Forecasting"** [cite: 3][cite_start], submitted by Usama Abdul Samad [cite: 5] [cite_start]in partial fulfillment of the requirements for the Master of Science in Data Analytics at the Universität Hildesheim[cite: 1, 13].

[cite_start]**Supervisors:** Prof. Dr. Dr. Lars Schmidt-Thieme[cite: 7, 8], M.Sc. [cite_start]Kiran Madhusudhanan [cite: 9, 10]

## Abstract

[cite_start]This thesis investigates the application of normalizing flows for multivariate time series forecasting, a critical task in many scientific and engineering domains[cite: 23]. [cite_start]The study explores deep learning models that combine the forecasting strength of autoregressive models (like RNNs and Transformers) with the flexibility of normalizing flows to represent complex data distributions[cite: 26].

[cite_start]A primary focus of this research is to empirically validate the effectiveness of the **Shiesh activation function** within the coupling layers of normalizing flows[cite: 28, 30]. [cite_start]Unlike conventional activation functions such as ReLU or Tanh, Shiesh is fully invertible, spans the entire real line, and maintains non-zero gradients, making it theoretically ideal for flow-based models[cite: 29, 395, 405].

## Key Contributions

* [cite_start]An empirical study on using normalizing flows conditioned on RNN and Transformer backbones for multivariate time series forecasting[cite: 1019].
* [cite_start]A practical investigation into the **Shiesh activation function** [cite: 590] [cite_start]to assess its impact on model performance compared to its theoretical advantages[cite: 991, 1025].
* [cite_start]Demonstration that extensive hyperparameter tuning of baseline models leads to significant performance improvements, with the tuned **Transformer-MAF** model outperforming the TimeGrad model[cite: 907, 980, 981].
* [cite_start]A comparative analysis using the **Solar** and **Electricity** datasets [cite: 807][cite_start], with performance evaluated using the Continuous Ranked Probability Score (CRPS) and Negative Log-Likelihood (NLL) metrics[cite: 834].

## Models and Architectures

The experiments in this thesis evaluate and compare the following architectures:
* GRU-RealNVP
* GRU-MAF
* Transformer-MAF
* Shiesh-Enhanced versions of the models above

## How to Cite

If you use the work or code from this repository, please consider citing the thesis.
