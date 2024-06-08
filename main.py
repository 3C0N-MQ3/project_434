# %% [markdown]
"""
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
# %% [markdown]
"""
<!--*!{misc/title.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/general.md}-->
"""
# %%
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
# %% [markdown]
"""
<!--*!{sections/q1.html}-->
"""
# %%
# Convert 'dateSurvey' to datetime format
data['dateSurvey'] = pd.to_datetime(data['dateSurvey'], errors='coerce')

# Set the index to be a MultiIndex for panel data
data = data.set_index(['agency_city', 'dateSurvey'])

# Define the dependent variable and independent variables
Y = np.log(data['UPTTotal'])
D = data['treatUberX']
W = data[['popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']]
PxD = data['PxD']
FxD = data['FxD']

# Scale the independent variables with log transformation
W_scaled_df = np.log(W)

# Create the design matrices
X = pd.concat([D, W_scaled_df], axis=1)

# Add constant to the models
X = sm.add_constant(X)

# %%
# Fit the OLS model
model1 = sm.OLS(Y, X).fit()

# Print the results
print(model1.summary())
# %% [markdown]
"""
<!--*!{sections/q2.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q3.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q4.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q5.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q6.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q7.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q8.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q9.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q10.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q11.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q12.html}-->
"""