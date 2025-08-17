# %%
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
import random


# %%
import numpy as np
import pandas as pd
import torch
import sys
import pts.dataset
#sys.path.append(r'C:\Users\usama\Desktop\Thesis\pytorch-ts-master')

# %%
#Notes Regarding versions:
#3.6.13 python --> 3.8.10
#0.10.0 gluonts --> 0.9.0
#Version: 1.1.5 pandas --> 1.5.3
#Version: 1.8.0 torch --> 2.1.5 --> 1.10.0 last version according to the issue answer in git
#Version: 1.19.5 numpy --> 1.23.5
#after this had issue regarding the distribution output importing changed the importing in the pts/dis_output line 34 to gluonts.torch.modules.distribution_output
# and remove it from the importing cell.
#encountered prefetch error handled it by downgrading torch version
#to install gluonts 0.9.0 had to downgrade pip version to 24.0
#!pip install optuna

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

# %%
dataset = get_dataset("solar_nips", regenerate=False)
dataset.metadata

# %%
train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                    max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
# val_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
# test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(train_data)), 
#                                    max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

# %%
#dataset_train = train_grouper(train_dataset)
dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)
# dataset_val = val_grouper(validation_dataset)

# %%
evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})

# %%
import numpy as np
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

# Define the grid for hyperparameter tuning
learning_rates = [0.001]  # Example learning rates
n_blocks_list = [8]  # Number of blocks to try

# Initialize variables to store the best results
best_metric = float('inf')  # Smallest CRPS-Sum
best_params = None

# Multivariate evaluator for evaluation
evaluator = MultivariateEvaluator(
    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
)

# Iterate over all combinations of hyperparameters
for learning_rate in learning_rates:
    for n_blocks in n_blocks_list:
        print(f"Trying learning_rate={learning_rate}, n_blocks={n_blocks}")
        
        # Define the estimator
        estimator = TempFlowEstimator(
            target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
            prediction_length=dataset.metadata.prediction_length,
            cell_type='GRU',
            input_size=552,
            freq=dataset.metadata.freq,
            scaling=True,
            dequantize=True,
            flow_type='MAF',
            n_blocks=n_blocks,
            trainer=Trainer(
                device=device,
                epochs=40,
                learning_rate=learning_rate,
                num_batches_per_epoch=100,
                batch_size=64,
            ),
        )
        
        # Train and evaluate the model
        predictor = estimator.train(dataset_train, num_workers=4)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset_test, predictor=predictor, num_samples=100
        )
        forecasts = list(forecast_it)
        targets = list(ts_it)
        
        # Evaluate the forecasts
        agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))
        current_metric = agg_metric['m_sum_mean_wQuantileLoss']
        
        print(f"CRPS-Sum for learning_rate={learning_rate}, n_blocks={n_blocks}: {current_metric}")
        
        # Update the best parameters if the current metric is better
        if current_metric < best_metric:
            best_metric = current_metric
            best_params = {
                'learning_rate': learning_rate,
                'n_blocks': n_blocks,
            }
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
# Print the best results
print(f"Best CRPS-Sum: {best_metric}")
print(f"Best Hyperparameters: {best_params}")



# %%



