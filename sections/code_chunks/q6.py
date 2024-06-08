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