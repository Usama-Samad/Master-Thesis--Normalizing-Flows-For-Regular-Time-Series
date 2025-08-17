# Normalizing Flows for Regular Time Series Forecasting

This repository contains the official source code, experiments, and final paper for the thesis titled **"Normalizing Flows for Regular Time Series Forecasting"**, submitted by Usama Abdul Samad in partial fulfillment of the requirements for the Master of Science in Data Analytics at the Universität Hildesheim.

**Supervisors:** Prof. Dr. Dr. Lars Schmidt-Thieme, M.Sc. Kiran Madhusudhanan

## Abstract

This thesis investigates the application of normalizing flows for multivariate time series forecasting, a critical task in many scientific and engineering domains. The study explores deep learning models that combine the forecasting strength of autoregressive models (like RNNs and Transformers) with the flexibility of normalizing flows to represent complex data distributions.

A primary focus of this research is to empirically validate the effectiveness of the **Shiesh activation function** within the coupling layers of normalizing flows. Unlike conventional activation functions such as ReLU or Tanh, Shiesh is fully invertible, spans the entire real line, and maintains non-zero gradients, making it theoretically ideal for flow-based models.

## Key Contributions

* An empirical study on using normalizing flows conditioned on RNN and Transformer backbones for multivariate time series forecasting.
* A practical investigation into the **Shiesh activation function** to assess its impact on model performance compared to its theoretical advantages.
* Demonstration that extensive hyperparameter tuning of baseline models leads to significant performance improvements, with the tuned **Transformer-MAF** model outperforming the TimeGrad model.
* A comparative analysis using the **Solar** and **Electricity** datasets, with performance evaluated using the Continuous Ranked Probability Score (CRPS) and Negative Log-Likelihood (NLL) metrics.

## Models and Architectures

The experiments in this thesis evaluate and compare the following architectures:

* GRU-RealNVP
* GRU-MAF
* Transformer-MAF
* Shiesh-Enhanced versions of the models above

## How to Cite

If you use the work or code from this repository, please consider citing the thesis.
