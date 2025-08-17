#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


# In[2]:


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


# In[3]:


import numpy as np
import pandas as pd
import random
import torch
import sys
import pts.dataset
#sys.path.append(r'C:\Users\usama\Desktop\Thesis\pytorch-ts-master')

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


dataset = get_dataset("electricity_nips", regenerate=False)
dataset.metadata


# In[8]:


#for electricity data
from gluonts.dataset.common import ListDataset

# Define train and validation datasets separately
train_data = []
val_data = []

for i, entry in enumerate(dataset.train):
    series_length = len(entry['target'])
    train_size = int(0.8 * series_length)  # 80% of the actual series length
    val_size = series_length - train_size  # Remaining 20%
    
    # Create train entry
    train_entry = entry.copy()
    train_entry['target'] = entry['target'][:train_size]  # First 80%
    train_data.append(train_entry)
    
    # Create validation entry
    val_entry = entry.copy()
    val_entry['target'] = entry['target'][train_size:]  # Remaining 20%
    val_data.append(val_entry)
    
    # Print verification
    print(f"Series {i}: Train length = {train_size}, Validation length = {val_size}")
    
    # Assert the lengths are correct
    assert len(train_entry['target']) == train_size, f"Train length mismatch for series {i}"
    assert len(val_entry['target']) == val_size, f"Validation length mismatch for series {i}"


# In[9]:


train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
val_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(train_data)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))


# In[10]:


train_dataset = ListDataset(train_data, freq=dataset.metadata.freq)
validation_dataset = ListDataset(val_data, freq=dataset.metadata.freq)


# In[11]:


dataset_train = train_grouper(train_dataset)
#dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)
dataset_val = val_grouper(validation_dataset)


# In[12]:


evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})


# In[13]:


import optuna
import torch

# Objective function for Optuna
def objective(trial):
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e-1)
    #batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    #epochs = trial.suggest_int("epochs", 25, 75)
    n_blocks = trial.suggest_int("n_blocks", 2, 8)
    
    # Initialize the Trainer with suggested parameters
    trainer = Trainer(
        device=device,
        epochs=45,
        learning_rate=learning_rate,
        batch_size=64,
        num_batches_per_epoch=100
    )
    
    # Initialize the Estimator with suggested parameters
    estimator = TempFlowEstimator(
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=dataset.metadata.prediction_length,
        cell_type='GRU',
        input_size=1484,
        freq=dataset.metadata.freq,
        scaling=True,
        dequantize=True,
        n_blocks=n_blocks,
        trainer=trainer
    )
    
    # Train the model and get the predictor
    predictor = estimator.train(dataset_train, dataset_val, num_workers=4)
    
    # Return the validation loss for the current trial (assuming Trainer has avg_val_loss attribute)
    return trainer.avg_val_loss

# Run the Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# Output best parameters and validation loss
print("Best parameters found:", study.best_params)
print("Best validation loss:", study.best_value)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




