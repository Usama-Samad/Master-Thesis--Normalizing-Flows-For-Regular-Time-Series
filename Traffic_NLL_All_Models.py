# %%
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import sys
import pts.dataset
import random

# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set your desired seed
set_seed(100)

# %%
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
#import gluonts.torch.distributions.distribution_output
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Prepeare data set

# %%
dataset = get_dataset("traffic_nips", regenerate=False)

# %%
dataset.metadata

# %%
train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

# %%
dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)

# %% [markdown]
# ## Evaluator

# %%
evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})

# %% [markdown]
# ## `GRU-Real-NVP`

# %%
estimator = TempFlowEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    cell_type='GRU',
    input_size=3856,
    freq=dataset.metadata.freq,
    scaling=True,
    dequantize=True,
    n_blocks=3,
    trainer=Trainer(device=device,
                    epochs=40,
                    learning_rate=1e-07,
                    num_batches_per_epoch=100,
                    batch_size=64)
)

# %%
predictor = estimator.train(dataset_train, num_workers=4)
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

# %% [markdown]
# ### Metrics

# %%
total_nll = 0.0
count = 0

for forecast, target in zip(forecasts, targets):
    # Compute mean and std from the forecast samples
    mu = forecast.samples.mean(axis=0)  # Mean of the samples
    sigma = forecast.samples.std(axis=0)  # Standard deviation of the samples

    # Align target to match the forecast's prediction length
    target = np.array(target[-forecast.samples.shape[1]:])  # Take the last `prediction_length` values

    if target.shape != mu.shape:
        print(f"Shape mismatch after alignment: target {target.shape}, forecast {mu.shape}")
        continue

    # Compute NLL for this time series
    nll = -torch.distributions.Normal(
        torch.tensor(mu), torch.tensor(sigma)
    ).log_prob(torch.tensor(target)).sum().item()

    total_nll += nll
    count += target.size  # Number of data points

# Compute mean NLL
if count > 0:
    mean_nll = total_nll / count
    print(f"Mean NLL on Test Set: {mean_nll}")
else:
    print("No valid data points for NLL calculation.")

# %%
print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))

# %%
print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))
print("Real NVP_ DONE#####################################")
# %% [markdown]
# ## `Transformer-MAF`

# %%
estimator = TransformerTempFlowEstimator(
    d_model=16,
    num_heads=4,
    input_size=3856,
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    context_length=dataset.metadata.prediction_length*4,
    flow_type='MAF',
    dequantize=True,
    freq=dataset.metadata.freq,
    trainer=Trainer(
        device=device,
        epochs=40,
        learning_rate=0.001,
        num_batches_per_epoch=100,
        batch_size=64,
    )
)

# %%
predictor = estimator.train(dataset_train, num_workers= 4)
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

# %% [markdown]
# ### Metrics

# %%
total_nll = 0.0
count = 0

for forecast, target in zip(forecasts, targets):
    # Compute mean and std from the forecast samples
    mu = forecast.samples.mean(axis=0)  # Mean of the samples
    sigma = forecast.samples.std(axis=0)  # Standard deviation of the samples

    # Align target to match the forecast's prediction length
    target = np.array(target[-forecast.samples.shape[1]:])  # Take the last `prediction_length` values

    if target.shape != mu.shape:
        print(f"Shape mismatch after alignment: target {target.shape}, forecast {mu.shape}")
        continue

    # Compute NLL for this time series
    nll = -torch.distributions.Normal(
        torch.tensor(mu), torch.tensor(sigma)
    ).log_prob(torch.tensor(target)).sum().item()

    total_nll += nll
    count += target.size  # Number of data points

# Compute mean NLL
if count > 0:
    mean_nll = total_nll / count
    print(f"Mean NLL on Test Set: {mean_nll}")
else:
    print("No valid data points for NLL calculation.")

# %%
print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))

# %%
print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))
print("TRANSFORMER_MAF_ DONE############################3")
# %%
# %% [markdown]
# ## `GRU-MAF`

# %%
estimator = TempFlowEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    cell_type='GRU',
    input_size=3856,
    freq=dataset.metadata.freq,
    scaling=True,
    dequantize=True,
    flow_type='MAF',
    n_blocks=6,
    trainer=Trainer(device=device,
                    epochs=40,
                    learning_rate=0.0001,
                    num_batches_per_epoch=100,
                    batch_size=64)
)

# %%
predictor = estimator.train(dataset_train, num_workers= 4)
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

# %% [markdown]
# ### Metrics

# %%
total_nll = 0.0
count = 0

for forecast, target in zip(forecasts, targets):
    # Compute mean and std from the forecast samples
    mu = forecast.samples.mean(axis=0)  # Mean of the samples
    sigma = forecast.samples.std(axis=0)  # Standard deviation of the samples

    # Align target to match the forecast's prediction length
    target = np.array(target[-forecast.samples.shape[1]:])  # Take the last `prediction_length` values

    if target.shape != mu.shape:
        print(f"Shape mismatch after alignment: target {target.shape}, forecast {mu.shape}")
        continue

    # Compute NLL for this time series
    nll = -torch.distributions.Normal(
        torch.tensor(mu), torch.tensor(sigma)
    ).log_prob(torch.tensor(target)).sum().item()

    total_nll += nll
    count += target.size  # Number of data points

# Compute mean NLL
if count > 0:
    mean_nll = total_nll / count
    print(f"Mean NLL on Test Set: {mean_nll}")
else:
    print("No valid data points for NLL calculation.")

# %%
print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))

# %%
print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))
print("MAF_ DONE############################3")




