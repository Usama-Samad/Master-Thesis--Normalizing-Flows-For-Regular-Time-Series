#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


# In[4]:


import numpy as np
import pandas as pd
import torch
import sys
import pts.dataset
#sys.path.append(r'C:\Users\usama\Desktop\Thesis\pytorch-ts-master')


# In[5]:


from gluonts.dataset.multivariate_grouper import MultivariateGrouper
#import gluonts.torch.distributions.distribution_output
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


dataset = get_dataset("solar_nips", regenerate=False)
dataset.metadata
train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))


# In[8]:


dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)


# In[9]:


evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})


# ### Sheish-Linear-Real-NVP

# In[10]:


#Best CRPS-Sum: 0.34631296069811673
#Best Hyperparameters: {'learning_rate': 0.00015417276294727372, 'batch_size': 128, 'epochs': 55}

import optuna

def objective(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-8, 1e-1)
    #batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    #epochs = trial.suggest_int('epochs', 20, 100)
    n_blocks = trial.suggest_int('n_blocks', 1, 10)
    #input_size = trial.suggest_int('input_size',)
    evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})
    estimator = TempFlowEstimator(
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=24,
        cell_type='GRU',
        input_size=552,
        freq=dataset.metadata.freq,
        scaling=True,
        dequantize=True,
        n_blocks=n_blocks,
        trainer=Trainer(
            device=device,
            epochs=45,
            learning_rate=learning_rate,
            num_batches_per_epoch=100,
            batch_size=64
        )
    )

    # Train and evaluate
    predictor = estimator.train(dataset_train, num_workers=4)
    forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                                     predictor=predictor,
                                                     num_samples=100)
    forecasts = list(forecast_it)
    targets = list(ts_it)
    
#    evaluator = Evaluator()
    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))
    
    # Objective to minimize
    return agg_metric['m_sum_mean_wQuantileLoss']

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # Adjust n_trials as needed

# Output the best trial
best_trial = study.best_trial
print(f"Best CRPS-Sum: {best_trial.value}")
print(f"Best Hyperparameters: {best_trial.params}")


# In[ ]:




