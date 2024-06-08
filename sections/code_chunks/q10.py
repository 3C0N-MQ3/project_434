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