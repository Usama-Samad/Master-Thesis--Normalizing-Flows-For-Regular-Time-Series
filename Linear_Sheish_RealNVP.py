#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


# In[1]:


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


# In[2]:


import numpy as np
import pandas as pd
import torch
import sys
import pts.dataset
#sys.path.append(r'C:\Users\usama\Desktop\Thesis\pytorch-ts-master')


# In[3]:


from gluonts.dataset.multivariate_grouper import MultivariateGrouper
#import gluonts.torch.distributions.distribution_output
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


dataset = get_dataset("solar_nips", regenerate=False)
dataset.metadata


# In[6]:


train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))


# In[7]:


dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)


# In[8]:


evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})


# In[9]:


estimator = TempFlowEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=24,
    cell_type='GRU',
    input_size=552,
    freq=dataset.metadata.freq,
    scaling=True,
    dequantize=True,
    n_blocks=4,
    trainer=Trainer(device=device,
                    epochs=45,
                    learning_rate=1.5170715286422367e-08,
                    num_batches_per_epoch=100,
                    batch_size=64)
)


# In[10]:


predictor = estimator.train(dataset_train, num_workers= 4)
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)
agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))


# In[ ]:


print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))


# In[ ]:


print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))

