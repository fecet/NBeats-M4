from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH=Path("./data/Dataset")

info_df=pd.read_csv(DATA_PATH/'M4-info.csv')

@dataclass()
class M4Meta:
    ids=info_df.M4id.values
    groups=info_df.SP.values
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }
    iterations = {
        'Yearly': 15000,
        'Quarterly': 15000,
        'Monthly': 15000,
        'Weekly': 5000,
        'Daily': 5000,
        'Hourly': 5000
    }


def read_data(freq):
    filename_train = DATA_PATH/f'Train/{freq}-train.csv'
    filename_test  = DATA_PATH/f'Test/{freq}-test.csv'
    df=pd.read_csv(filename_train)
    tss=df.drop('V1',axis=1).values.copy(order='C').astype(np.float32)
    def dropna(x):
        return x[~np.isnan(x)]

    timeseries=[dropna(ts) for ts in tss]
    df=pd.read_csv(filename_test)
    targets=df.drop('V1',axis=1).values.copy(order='C').astype(np.float32)
    return timeseries,targets
