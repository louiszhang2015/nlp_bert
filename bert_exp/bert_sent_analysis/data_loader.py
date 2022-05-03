import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def load_train_data():

    data_complaint = pd.read_csv('data/sentiment_analysis/complaint1700.csv')
    data_complaint['label'] = 0
    data_non_complaint = pd.read_csv('data/sentiment_analysis/noncomplaint1700.csv')
    data_non_complaint['label'] = 1

    # Concatenate complaining and non-complaining data
    data = pd.concat([data_complaint, data_non_complaint], axis=0).reset_index(drop=True)

    # Drop 'airline' column
    data.drop(['airline'], inplace=True, axis=1)

    return data


def load_test_data():


    test_data = pd.read_csv('data/sentiment_analysis/test_data.csv')

    # Keep important columns
    test_data = test_data[['id', 'tweet']]

    return test_data


def torch_data_loader(train_inputs, train_masks, val_inputs, val_masks, y_train, y_val):

    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
    batch_size = 32

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader
