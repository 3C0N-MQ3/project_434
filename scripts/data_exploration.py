# %%
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from toolz import pipe
from rich import inspect
# %%
data = pipe(
    '../data/uber_dataset.csv',
    lambda x: pd.read_csv(x),
    lambda x: x.drop(columns=['Unnamed: 0']),
    lambda x: x.drop(columns=['treatGTNotStd']),
    lambda x: x.dropna()
)

data['agency'] = data['agency'].astype('category')
data['city'] = data['city'].astype('category')
data['state'] = data['state'].astype('category')
data['dateSurvey'] = pd.to_datetime(data['dateSurvey'])

data.set_index(['agency', 'dateSurvey'], inplace=True)
# %%
data