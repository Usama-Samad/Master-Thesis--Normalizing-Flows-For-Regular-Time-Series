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


# In[4]:


from gluonts.dataset.multivariate_grouper import MultivariateGrouper
#import gluonts.torch.distributions.distribution_output
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:


dataset = get_dataset("solar_nips", regenerate=False)
dataset.metadata


# In[7]:


from gluonts.dataset.common import ListDataset
train_size = int(0.8 * 7009)  # 80% of 7009 for training

# Define train and validation datasets separately
train_data = []
val_data = []

for entry in dataset.train:
    # Create train entry
    train_entry = entry.copy()
    train_entry['target'] = entry['target'][:train_size]  # First 5607 values
    train_data.append(train_entry)
    
    # Create validation entry
    val_entry = entry.copy()
    val_entry['target'] = entry['target'][train_size:]  # Remaining 1402 values
    val_data.append(val_entry)

# Verify that train and validation sets have the expected number of time steps
for i, (train_entry, val_entry) in enumerate(zip(train_data, val_data)):
    train_length = len(train_entry['target'])
    val_length = len(val_entry['target'])
    
    print(f"Series {i}: Train length = {train_length}, Validation length = {val_length}")
    
    # Optional: Assert that lengths are correct
    assert train_length == 5607, f"Train length mismatch for series {i}: got {train_length}"
    assert val_length == 1402, f"Validation length mismatch for series {i}: got {val_length}"


# In[8]:


train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
val_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(train_data)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))


# In[9]:


dataset_train = train_grouper(train_data)
dataset_test = test_grouper(dataset.test)
dataset_val = val_grouper(val_data)


# In[10]:


evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})


# In[ ]:


import optuna
import torch

# Objective function for Optuna
def objective(trial):
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-1)
    #batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    #epochs = trial.suggest_int("epochs", 25, 75)
    n_blocks = trial.suggest_int("n_blocks", 2, 6)
    
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
        input_size=552,
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
study.optimize(objective, n_trials=100)

# Output best parameters and validation loss
print("Best parameters found:", study.best_params)
print("Best validation loss:", study.best_value)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




