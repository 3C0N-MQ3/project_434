# %% [markdown]
# University of California Los Angeles  
# Master of Quantitative Economics -MQE-  
# ECON-434-Machine Learning and Big Data for Economists
# 
# <p style='text-align: right;'>Luis Alejandro Samayoa Alvarado </p>
# <p style='text-align: right;'>UID 506140191</p>

# %% [markdown]
# <div style="text-align: center;">
# <h1>Homework No.3</h1>
# </div>

# %%
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Problem 1
# Provide a python code to calculate the double Lasso estimator as well as the corresponding asymptotic standard errors.
# 
# * *First, I will define a function to estimate the lambda using the BCCH method.*

# %%
# Define BCCH function
def BCCH(X, Y, c=1.1, alpha=0.05):
    X_copy = np.array(X)
    Y_copy = np.array(Y)

    if len(X_copy) != len(Y_copy):
        raise ValueError("Length of X and Y must be the same")

    n, p = X_copy.shape

    Y_reshaped = np.tile(Y_copy, (p, 1)).T
    maximum = np.max((np.mean((X_copy * Y_reshaped) ** 2, axis=1)) ** 0.5)
    ppf = norm.ppf(1 - alpha / (2 * p))
    lambda_pilot = (c / (n ** 0.5)) * ppf * maximum
    lasso = Lasso(alpha=lambda_pilot)                                 
    lasso.fit(X_copy, Y_copy)
    Y_prediction = lasso.predict(X_copy).reshape(-1, 1)
    e = Y_copy - Y_prediction.flatten()
    e_reshaped = np.tile(e, (p, 1)).T
    new_maximum = np.max((np.mean((X_copy * e_reshaped) ** 2, axis=1)) ** 0.5)
    lambda_final = (c / (n ** 0.5)) * ppf * new_maximum
    return lambda_final

# %% [markdown]
# ### Traditional way
# 
# * *Steps to perform double lasso following the slides instructions.*
# * *Estimation of alpha and estandar desviation using formulas in slides.*

# %%
def double_lasso(Y, D, Z):
    # Step 1: LASSO of Y on D and Z
    X_y = np.concatenate((D.reshape(-1, 1), Z), axis=1)
    bcch_y = BCCH(X_y, Y)
    lasso_y = Lasso(alpha=bcch_y)
    lasso_y.fit(X_y, Y)
    gammas_hat = lasso_y.coef_[1:]
    predicted_y = lasso_y.predict(X_y)
    e_residuals = Y - predicted_y

    # Step 2: LASSO of D on Z
    bcch_d = BCCH(Z, D)
    lasso_d = Lasso(alpha=bcch_d)
    lasso_d.fit(Z, D)
    predicted_d = lasso_d.predict(Z)
    v_residuals = D - predicted_d
    
    # Step 3: Estimate alpha
    seudo_residuals = Y - np.dot(Z, gammas_hat) # Y - Z*gammas_hat
    numerator = np.sum(seudo_residuals * v_residuals)
    denominator = np.sum(D * v_residuals)
    estimated_alpha = numerator / denominator

    # Calculate estimated standard error
    var_num = np.mean((e_residuals * v_residuals)** 2)
    var_den = (np.mean(v_residuals**2))**2
    
    var_error = var_num / var_den
    std_error = np.sqrt(var_error/len(Y))
    return estimated_alpha, std_error

# Example usage:
# Generate synthetic data
np.random.seed(150)
n = 1000
p = 10

D = np.random.randn(n)
Z = np.random.randn(n, p)
Y = 0.5 * D + np.dot(Z, np.ones(p)) + np.random.randn(n)

# Run double LASSO
estimated_alpha, estimated_std_error = double_lasso(Y, D, Z)
print("Estimated alpha:", estimated_alpha.round(4))
print("Estimated standard error:", estimated_std_error.round(4))
min = estimated_alpha - 1.96 * estimated_std_error
max = estimated_alpha + 1.96 * estimated_std_error
print("Confidence interval:", (min.round(4), max.round(4)))

# %% [markdown]
# #### Alternative Implementation
# 
# * *Compare with the alternative implementation, using an Ordinary Least Square in the last step to obtain the standard errors.*

# %%
def double_lasso_OLS(Y, D, Z):
    # Step 1: LASSO of Y on D and Z
    X = np.concatenate((D.reshape(-1, 1), Z), axis=1)
    lambda_y = BCCH(X, Y)
    lasso_y = Lasso(alpha=lambda_y)
    lasso_y.fit(X, Y)
    selected_variables_y = X[:, lasso_y.coef_ != 0]

    # Step 2: LASSO of D on Z
    D_lasso = D.flatten()
    lambda_d = BCCH(Z, D_lasso)
    lasso_d = Lasso(alpha=lambda_d)
    lasso_d.fit(Z, D_lasso)
    selected_variables_d = Z[:, lasso_d.coef_ != 0]

    # Step 3: OLS of Y on D and selected variables from steps 1 and 2
    D_reshaped = D.reshape(-1, 1)
    X_selected = np.concatenate((D_reshaped, selected_variables_y, selected_variables_d), axis=1)
    ols = LinearRegression()
    ols.fit(X_selected, Y)
    
    # Get standard errors of the coefficients
    residuals = Y - ols.predict(X_selected)
    mse_residuals = np.mean(residuals ** 2)
    X_selected_T = X_selected.T
    var_cov_matrix = mse_residuals * np.linalg.inv(np.dot(X_selected_T, X_selected))
    std_errors = np.sqrt(np.diag(var_cov_matrix))
    
    # Print coefficient alpha, standard error, and confidence interval
    coefficient_alpha = ols.coef_[0]
    standard_error_alpha = std_errors[0]
    ci_min = coefficient_alpha - 1.96 * standard_error_alpha
    ci_max = coefficient_alpha + 1.96 * standard_error_alpha
    
    return coefficient_alpha, standard_error_alpha, (ci_min, ci_max)

# Example usage:

# Generate synthetic data
np.random.seed(150)
n = 1000
p = 10
D = np.random.randn(n)
Z = np.random.randn(n, p)
Y = 0.5 * D + np.dot(Z, np.ones(p)) + np.random.randn(n)

# Run double LASSO regression
estimated_alpha, estimated_std_error, ci_alpha = double_lasso_OLS(Y, D, Z)
print("Estimated alpha:", estimated_alpha.round(4))
print("Estimated standard error of alpha:", estimated_std_error.round(4))
print("95% Confidence interval of alpha:", (ci_alpha[0].round(4), ci_alpha[1].round(4)))

# %% [markdown]
# * *As we can see, both methods to do Double Lasso returns the same estimated alpha and almost the same estandard error.*

# %% [markdown]
# The dataset includes information on both the transit agencies and on the Metropolitan Statistical Areas (MSA) where they operate. For each time period, the dataset contains values for the following variables:
# 
# 1. UPTTotal – the number of rides for the public transit agency;
# 2. treatUberX - a dummy for Uber presence in the corresponding MSA;
# 3. treatGTNotStd - a variable measuring google search intensity for Uber in the corresponding MSA;
# 4. popestimate - population in the corresponding MSA;
# 5. employment - employment in the corresponding MSA;
# 6. aveFareTotal - average fare for the public transit agency;
# 7. VRHTTotal - vehicle hours for the public transit agency;
# 8. VOMSTotal - number of vehicles employed by the public transit agency;
# 9. VRMTotal - vehicle miles for the public transit agency;
# 10. gasPrice - gas price in the corresponding MSA.
# 
# In this dataset, treatUber and treatGTNotStd is qualitative and quantitative measures for the same thing: Uber presense in the MSA. We can run regressions using either of these two variables and then check whether results are robust if the other variable is used.
# 
# There are two variations in this dataset that allow us to study the effect of Uber on public transit. First, in any given time period, Uber is present in some MSAs but not in others. We can thus study the effect of Uber by comparing these MSAs. Second, for any given MSA, we have data on time periods both before and after Uber was introduced in this MSA. We can thus study the effect 1of Uber by comparing these time periods. By working with panel data, we are able to employ both variations at the same time.
# 
# To study the effect of Uber on public transit, we let Yit be UPTTotal, Dit be either treatUberX or treatGTNotStd, and Wit be the vector including remaining variables: popestimate, employment, aveFareTotal, VRHTTotal, VOMSTotal, VRMTotal, gasPrice. We then run the following regressions:

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# %%
# Load data
data = pd.read_csv("uber_dataset.csv", index_col=0)

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
# 1. $OLS: log Y_{it} = \alpha + D_{it}\beta + W_{it}\gamma + e_{it}$

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
X1 = pd.concat([D, W_scaled_df], axis=1)

# Add constant to the models
X1 = sm.add_constant(X1)

# %%
# Fit the OLS model
model1 = sm.OLS(Y, X1).fit()

# Print the results
print(model1.summary())

# %% [markdown]
# 2. $OLS: log Y_{it} = \eta_i + \delta_t + D_{it}\beta + W_{it}\gamma + e_{it}$

# %%
# Ensure Y is a Series rather than a DataFrame
Y = Y.squeeze()

# Create the design matrices
X2 = pd.concat([D, W_scaled_df], axis=1)

# Fit the Panel OLS models with individual and time fixed effects
model2 = PanelOLS(Y, X2, entity_effects=True, time_effects=True, drop_absorbed=True)
result2 = model2.fit()

# Print the summaries to check the fixed effects inclusion
print(result2.summary)

# %% [markdown]
# 3. $OLS: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}P_{it}\beta_{2} + W_{it}\gamma + e_{it}$; where $P_{it}$ is a dummy that takes value 1 if the corresponding MSA has population larger than the median population in the dataset and 0 otherwise.

# %%
# Create the design matrices
X3 = pd.concat([D, PxD, W], axis=1)

# Fit the Panel OLS models with individual and time fixed effects
model3 = PanelOLS(Y, X3, entity_effects=True, time_effects=True, drop_absorbed=True)
result3 = model3.fit()

print(result3.summary)

# %% [markdown]
# 4. $OLS: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}F_{it}\beta_{2} + W_{it}\gamma + e_{it}$; where $F_{it}$ is a dummy that takes value 1 if the number of rides of the public travel agency is larger than the median number of rides among all public transit agencies in the dataset.

# %%
# Create the design matrices
X4 = pd.concat([D, FxD, W], axis=1)

# Fit the Panel OLS models with individual and time fixed effects
model4 = PanelOLS(Y, X4, entity_effects=True, time_effects=True, drop_absorbed=True)
result4 = model4.fit()

print(result4.summary)

# %% [markdown]
# 5. $LASSO: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}P_{it}\beta_{2} + W_{it}\gamma + e_{it}$; where $P_{it}$ is a dummy that takes value 1 if the corresponding MSA has population larger than the median population in the dataset and 0 otherwise.

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

# Encode entity and time as dummy variables
entity_dummies = pd.get_dummies(data['agency_city'], drop_first=True)
time_dummies = pd.get_dummies(data['dateSurvey'], drop_first=True)

# Create the design matrices
X5 = np.column_stack((D, PxD, W_scaled_df, entity_dummies, time_dummies))
Y = np.log(data['UPTTotal'])

# %%
# Fit Lasso regression models
alpha1 = BCCH(X5, Y)
lasso5 = Lasso(alpha=alpha1)  # You can adjust the alpha parameter as needed
lasso5.fit(X5, Y)

# Define the feature names
feature_names = ['D', 'PD', 'popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']

# Create DataFrame for Model 1
coef5_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso5.coef_[:9]
})

print("Model 5 Coefficients:")
print(coef5_df)

# %% [markdown]
# 6. $LASSO: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}F_{it}\beta_{2} + W_{it}\gamma + e_{it}$; where $F_{it}$ is a dummy that takes value 1 if the number of rides of the public travel agency is larger than the median number of rides among all public transit agencies in the dataset.

# %%
# Create the design matrices
X6 = np.column_stack((D, FxD, W_scaled_df, entity_dummies, time_dummies))

# Fit Lasso regression models
alpha2 = BCCH(X6, Y)
lasso6 = Lasso(alpha=alpha2)  # You can adjust the alpha parameter as needed
lasso6.fit(X6, Y)

# Define the feature names
feature_names = ['D', 'FD', 'popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']

# Create DataFrame for Model 1
coef6_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso6.coef_[:9]
})

print("Model 6 Coefficients:")
print(coef6_df)


# %% [markdown]
# 7. $Double-LASSO: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}P_{it}\beta_{2} + W_{it}\gamma + e_{it}$; where $P_{it}$ is a dummy that takes value 1 if the corresponding MSA has population larger than the median population in the dataset and 0 otherwise.

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
# 8. $Double-LASSO: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}F_{it}\beta_{2} + W_{it}\gamma + e_{it}$; where $F_{it}$ is a dummy that takes value 1 if the number of rides of the public travel agency is larger than the median number of rides among all public transit agencies in the dataset.

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
# 9. $LASSO: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}P_{it}\beta_{2} + \tilde{W}_{it} \gamma + e_{it}$, where where coefficients of interest are $\beta_1$ and $\beta_2$ $\tilde{W}_{it}$ includes all interactions of order 5 of variables in the vector $W_{it}.$

# %%
# Create polynomial features of 5th order
poly = PolynomialFeatures(degree=5)
W_poly = poly.fit_transform(W_scaled_df)
W_p = pd.DataFrame(W_poly)
W_p.drop(columns= 0, inplace=True)

# %%
# Create the design matrices
X9 = np.column_stack((D, PxD, W_p, entity_dummies, time_dummies))

# Fit Lasso regression models
alpha3 = BCCH(X9, Y)
lasso9 = Lasso(alpha=alpha3)  # You can adjust the alpha parameter as needed
lasso9.fit(X9, Y)

# Define the feature names
feature_names = ['D', 'PD', 'popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']

# Create DataFrame for Model 9
coef9_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso9.coef_[:9]
})

print("Model 9 Coefficients:")
print(coef9_df)

# %% [markdown]
# 10. $LASSO: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}F_{it}\beta_{2} + \tilde{W}_{it} \gamma + e_{it}$, where where coefficients of interest are $\beta_1$ and $\beta_2$ $\tilde{W}_{it}$ includes all interactions of order 5 of variables in the vector $W_{it}.$

# %%
# Create the design matrices
X10 = np.column_stack((D, FxD, W_p, entity_dummies, time_dummies))

# Fit Lasso regression models
alpha4 = BCCH(X10, Y)
lasso10 = Lasso(alpha=alpha4)  # You can adjust the alpha parameter as needed
lasso10.fit(X10, Y)

# Define the feature names
feature_names = ['D', 'FD', 'popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']

# Create DataFrame for Model 10
coef10_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso10.coef_[:9]
})

print("Model 10 Coefficients:")
print(coef10_df)

# %% [markdown]
# 11. $Double-LASSO: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}P_{it}\beta_{2} + \tilde{W}_{it}\gamma + e_{it}$; where where coefficients of interest are $\beta_1$ and $\beta_2$ $\tilde{W}_{it}$ includes all interactions of order 5 of variables in the vector $W_{it}.$

# %%
Y = np.array(np.log(data['UPTTotal']), ndmin=1).T
D = np.array(data['treatUberX'], ndmin=1).T
W = np.array(np.log(data[['popestimate', 'employment', 'aveFareTotal', 'VRHTotal', 'VOMSTotal', 'VRMTotal', 'gasPrice']]))
P = np.array(data['P'], ndmin=1).T
W1 = np.column_stack((D*P, W_p))
W2 = np.column_stack((D, W_p))
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
# 12. $Double-LASSO: log Y_{it} = \eta_i + \delta_t + D_{it}\beta_{1} + D_{it}F_{it}\beta_{2} + \tilde{W}_{it}\gamma + e_{it}$; where where coefficients of interest are $\beta_1$ and $\beta_2$ $\tilde{W}_{it}$ includes all interactions of order 5 of variables in the vector $W_{it}.$

# %%
# Create Fit
F = np.array(data['F'], ndmin=1).T
W3 = np.column_stack((D*F, W_p))
DF = D * F

# Convert dummy variables to numpy arrays
entity_dummies_array = entity_dummies.to_numpy()
time_dummies_array = time_dummies.to_numpy()

# Concatenate the arrays
W3_combined = np.concatenate([W3, entity_dummies_array, time_dummies_array], axis=1)

# Run double LASSO regression to estimate alpha for D
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


