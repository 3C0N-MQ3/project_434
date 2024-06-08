# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %% [markdown]
"""
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
# %% [markdown]
# <div style="text-align: left;">
#     <h1>Uber Versus Public Transit</h1>
#     <h2>Final Project</h2>
#     <h4>ECON 434 - Machine Learning and Big Data for Economists</h3>
#     <div style="padding: 20px 0;">
#         <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));">
#         <p><em>Luis-Alejandro Samayoa-Alvarado</em><br>
#         <p><em>Mauricio Vargas-Estrada</em><br>
#         <p><em>Nikolas Papadatos</em><br>
#         <p><em>William Borelli Ebert</em><br>
#         <p>Master of Quantitative Economics<br>
#         University of California - Los Angeles</p>
#         <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));">
#     </div>
# </div>
# %% [markdown]
"""
In this project, we study whether Uber complements (helps) or substitutes (hurts) public transit. On the one hand, Uber can substitute public transit if riders decide to choose Uber instead of public transit. On the other hand, Uber can complement public transit if riders take Uber from home to public transit stop, which can make public transit more attractive than driving a car. The net effect is unclear and is a subject of intense policy debate.

We will expand on the original set of results presented in Hall, Palsson, and Price (2018), “Is Uber a substitute or a complement for public transit,” *Journal of Urban Economics*, which is available on the class website. We will use their dataset, which is also available on the class website. In the dataset, a unit of ob- servation is a public transit agency in a given year-month. The dataset includes information on both the transit agencies and on the Metropolitan Statistical Areas (MSA) where they operate. For each time period, the dataset contains values for the following variables:

1. $UPTTotal$ – the number of rides for the public transit agency;
2. $treatUberX$ - a dummy for Uber presence in the corresponding MSA;
3. $treatGTNotStd$ - a variable measuring google search intensity for Uber in the corresponding MSA;
4. $popestimate$ - population in the corresponding MSA;
5. $employment$ - employment in the corresponding MSA;
6. $aveFareTotal$ - average fare for the public transit agency;
7. $VRHTTotal$ - vehicle hours for the public transit agency;
8. $VOMSTotal$ - number of vehicles employed by the public transit agency;
9. $VRMTotal$ - vehicle miles for the public transit agency;
10. $gasPrice$ - gas price in the corresponding MSA.

In this dataset, $treatUberX$ and $treatGTNotStd$ is qualitative and quantitative measures for the same thing: Uber presence in the MSA. We can run regressions using either of these two variables and then check whether results are robust if the other variable is used.

There are two variations in this dataset that allow us to study the effect of Uber on public transit. First, in any given time period, Uber is present in some MSAs but not others. We can thus study the effect of Uber by comparing these MSAs. Second, for any given MSA, we have data on time periods both before and after Uber was introduced in this MSA. We can thus study the effect of Uber by comparing these time periods. By working with panel data, we are able to employ both variations at the same time.

To study the effect of Uber on public transit, we let $Y_{it}$ be $UPTTotal$, $D_{it}$ be either $treatUberX$ or $treatGTNotStd$, and $W_{it}$ be the vector including remaining variables: $popestimate$, $employment$, $aveFareTotal$, $VRHTTotal$, $VOMSTotal$, $VRMTotal$, $gasPrice$. We then run the following regressions:
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

# Create interaction between agency and city
data['agency_city'] = data['agency'] + data['city']

# Calculate the median population 
data_copy = data[['UPTTotal', 'popestimate', 'city']].copy()
p = data_copy.groupby(['city']).median()
median_population = p['popestimate'].median()

# Create the dummy variable P_{it}
data['P'] = (data['popestimate'] > median_population).astype(int)

# Calculate the median rides 
median_rides = p['UPTTotal'].median()

# Create the dummy variable F_{it}
data['F'] = (data['UPTTotal'] > median_rides).astype(int)

# Create the interaction term P_{it} * D_{it}
data['PxD'] = data['P'] * data['treatUberX']

# Create the interaction term F_{it} * D_{it}
data['FxD'] = data['F'] * data['treatUberX']
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         1.)
#     </div>
#     <div style="padding: 10px;">
#         OLS: log \(Y_{it} = \alpha + D_{it} \beta + W_{it}' \gamma + \epsilon_{it}\).
#     </div>
# </div>
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
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         2.)
#     </div>
#     <div style="padding: 10px;">
#         OLS: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta + W_{it}' \gamma + \epsilon_{it}\).
#     </div>
# </div>
# %%
# Ensure Y is a Series rather than a DataFrame
Y = Y.squeeze()

# Create the design matrices
X = pd.concat([D, W_scaled_df], axis=1)

# Fit the Panel OLS models with individual and time fixed effects
model2 = PanelOLS(Y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
result2 = model2.fit()

# Print the summaries to check the fixed effects inclusion
print(result2.summary)
# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         3.)
#     </div>
#     <div style="padding: 10px;">
#         OLS: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Pit_i \beta_2 + W_{it}' \gamma + \epsilon_{it}\), where \(Pit_i\) is a dummy that takes value 1 if the corresponding MSA has population larger than the median population in the dataset and 0 otherwise.
#     </div>
# </div>
# %%
# Create the design matrices
X1 = pd.concat([D, PxD, W], axis=1)

# Fit the Panel OLS models with individual and time fixed effects
model3 = PanelOLS(Y, X1, entity_effects=True, time_effects=True, drop_absorbed=True)
result3 = model3.fit()

print(result3.summary)
# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         4.)
#     </div>
#     <div style="padding: 10px;">
#         OLS: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Fit_i \beta_2 + W_{it}' \gamma + \epsilon_{it}\), where \(Fit_i\) is a dummy that takes value 1 if the number of rides of the public travel agency is larger than the median number of rides among all public transit agencies in the dataset.
#     </div>
# </div>
# %%
# Create the design matrices
X2 = pd.concat([D, FxD, W], axis=1)

# Fit the Panel OLS models with individual and time fixed effects
model4 = PanelOLS(Y, X2, entity_effects=True, time_effects=True, drop_absorbed=True)
result4 = model4.fit()

print(result4.summary)
# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         5.)
#     </div>
#     <div style="padding: 10px;">
#         LASSO: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Pit \beta_2 + W_{it}' \gamma + \epsilon_{it}\).
#     </div>
# </div>
# %%
# Load data
data.reset_index(inplace=True)

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

# Encode entity and time as dummy variables
entity_dummies = pd.get_dummies(data['agency_city'], drop_first=True)
time_dummies = pd.get_dummies(data['dateSurvey'], drop_first=True)

# Create the design matrices
X3 = np.column_stack((D, PxD, W_scaled_df, entity_dummies, time_dummies))
Y = np.log(data['UPTTotal'])

# %%
# Fit Lasso regression models
alpha1 = BCCH(X3, Y)
lasso1 = Lasso(alpha=alpha1)  # You can adjust the alpha parameter as needed
lasso1.fit(X3, Y)

# Define the feature names
feature_names = ['D', 'P', 'popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']

# Create DataFrame for Model 1
coef1_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso1.coef_[:9]
})

print("Model 1 Coefficients:")
print(coef1_df)
# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         6.)
#     </div>
#     <div style="padding: 10px;">
#         LASSO: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Fit \beta_2 + W_{it}' \gamma + \epsilon_{it}\).
#     </div>
# </div>
# %%
# Create the design matrices
X4 = np.column_stack((D, FxD, W_scaled_df, entity_dummies, time_dummies))

# Fit Lasso regression models
alpha2 = BCCH(X4, Y)
lasso2 = Lasso(alpha=alpha1)  # You can adjust the alpha parameter as needed
lasso2.fit(X4, Y)

# Define the feature names
feature_names = ['D', 'F', 'popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']

# Create DataFrame for Model 1
coef2_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso2.coef_[:9]
})

print("Model 2 Coefficients:")
print(coef2_df)
# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         7.)
#     </div>
#     <div style="padding: 10px;">
#         Double-LASSO: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Pit \beta_2 + \tilde{W}_{it}' \gamma + \epsilon_{it}\), where coefficients of interest are \(\beta_1\) and \(\beta_2\).
#     </div>
# </div>
# %%
Y = np.array(np.log(data['UPTTotal']), ndmin=1).T
D = np.array(data['treatUberX'], ndmin=1).T
W = np.array(np.log(data[['popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']]))
P = np.array(data['P'], ndmin=1).T
W1 = np.column_stack((D*P, W))
W2 = np.column_stack((D, W))
DP = D * P

# Convert dummy variables to numpy arrays
entity_dummies_array = entity_dummies.to_numpy()
time_dummies_array = time_dummies.to_numpy()

# Concatenate the arrays
W1_combined = np.concatenate([W1, entity_dummies_array, time_dummies_array], axis=1)
W2_combined = np.concatenate([W2, entity_dummies_array, time_dummies_array], axis=1)


# Run double LASSO regression to estimate alpha for D
estimated_alpha, estimated_std_error = double_lasso(Y, D, W1_combined)
print("Estimated alpha:", estimated_alpha.round(4))
print("Estimated standard error:", estimated_std_error.round(4))
min = estimated_alpha - 1.96 * estimated_std_error
max = estimated_alpha + 1.96 * estimated_std_error
print("Confidence interval:", (min.round(4), max.round(4)))
# %%
# Run double LASSO regression to estimate alpha for D*P
estimated_alpha, estimated_std_error = double_lasso(Y, DP, W2_combined)
print("Estimated alpha:", estimated_alpha.round(4))
print("Estimated standard error:", estimated_std_error.round(4))
min = estimated_alpha - 1.96 * estimated_std_error
max = estimated_alpha + 1.96 * estimated_std_error
print("Confidence interval:", (min.round(4), max.round(4)))
# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         8.)
#     </div>
#     <div style="padding: 10px;">
#         Double-LASSO: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Fit \beta_2 + \tilde{W}_{it}' \gamma + \epsilon_{it}\), where coefficients of interest are \(\beta_1\) and \(\beta_2\).
#     </div>
# </div>
# %%
F = np.array(data['F'], ndmin=1).T
W3 = np.column_stack((D*F, W))
W3_combined = np.concatenate([W3, entity_dummies_array, time_dummies_array], axis=1)
DF = D * F

# Run double LASSO regression to estimate alpha for D, using F as an instrument
estimated_alpha, estimated_std_error = double_lasso(Y, D, W3_combined)
print("Estimated alpha:", estimated_alpha.round(4))
print("Estimated standard error:", estimated_std_error.round(4))
min = estimated_alpha - 1.96 * estimated_std_error
max = estimated_alpha + 1.96 * estimated_std_error
print("Confidence interval:", (min.round(4), max.round(4)))

# %%
# Run double LASSO regression to estimate alpha for D*F
estimated_alpha, estimated_std_error = double_lasso(Y, DF, W2_combined)
print("Estimated alpha:", estimated_alpha.round(4))
print("Estimated standard error:", estimated_std_error.round(4))
min = estimated_alpha - 1.96 * estimated_std_error
max = estimated_alpha + 1.96 * estimated_std_error
print("Confidence interval:", (min.round(4), max.round(4)))
# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         9.)
#     </div>
#     <div style="padding: 10px;">
#         LASSO: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Pit \beta_2 + \tilde{W}_{it}' \gamma + \epsilon_{it}\), where \(\tilde{W}_{it}\) includes all interactions of order 5 of variables in the vector \(W_{it}\).
#     </div>
# </div>
# %%

# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         10.)
#     </div>
#     <div style="padding: 10px;">
#         LASSO: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Fit \beta_2 + \tilde{W}_{it}' \gamma + \epsilon_{it}\), where \(\tilde{W}_{it}\) includes all interactions of order 5 of variables in the vector \(W_{it}\).
#     </div>
# </div>
# %%

# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         11.)
#     </div>
#     <div style="padding: 10px;">
#         Double-LASSO: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Pit \beta_2 + \tilde{W}_{it}' \gamma + \epsilon_{it}\), where coefficients of interest are \(\beta_1\) and \(\beta_2\) and \(\tilde{W}_{it}\) includes all interactions of order 5 of variables in the vector \(W_{it}\).
#     </div>
# </div>
# %%

# %% [markdown]
#
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#         12.)
#     </div>
#     <div style="padding: 10px;">
#         Double-LASSO: log \(Y_{it} = \eta_i + \delta_t + D_{it} \beta_1 + D_{it} Fit \beta_2 + \tilde{W}_{it}' \gamma + \epsilon_{it}\), where coefficients of interest are \(\beta_1\) and \(\beta_2\) and \(\tilde{W}_{it}\) includes all interactions of order 5 of variables in the vector \(W_{it}\).
#     </div>
# </div>
# %%

# %% [markdown]
#
