import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import PanelOLS

from src.BCCH import BCCH
from src.double_lasso import double_lasso
from src.double_lasso_OLS import double_lasso_OLS
# %%
# Load data
data = pd.read_csv("data/uber_dataset.csv", index_col=0)

# Drop treatGTNotStd variable
data = data.drop(columns='treatGTNotStd')

# Drop rows with missing values
data = data.dropna()

# If treatUberX is greater than 0.5, set it to 1, if not, set it to 0
data['treatUberX'] = (data['treatUberX'] > 0.5).astype(int)

# Calculate the median population across all entities
median_population = data.groupby('dateSurvey')['popestimate'].median()

# Merge the median population back to the original dataframe
data = data.reset_index().merge(median_population.rename('median_pop'), on='dateSurvey')

# Create the dummy variable P_{it}
data['P'] = (data['popestimate'] > data['median_pop']).astype(int)

# Calculate the median rides across all times
median_rides = data.groupby('dateSurvey')['UPTTotal'].median()

# Merge the median population back to the original dataframe
data = data.reset_index().merge(median_rides.rename('median_ride'), on='dateSurvey')

# Create the dummy variable F_{it}
data['F'] = (data['UPTTotal'] > data['median_ride']).astype(int)

# Create the interaction term P_{it} * D_{it}
data['PxD'] = data['P'] * data['treatUberX']

# Create the interaction term F_{it} * D_{it}
data['FxD'] = data['F'] * data['treatUberX']

# Create interaction between agency and city
data['agency_city'] = data['agency'] + data['city']