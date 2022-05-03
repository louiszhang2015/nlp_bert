import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
