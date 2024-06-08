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