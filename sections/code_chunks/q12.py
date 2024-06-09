# Create Fit
F = np.array(data["F"], ndmin=1).T
W3 = np.column_stack((D * F, W_p))
DF = D * F

# Convert dummy variables to numpy arrays
entity_dummies_array = entity_dummies.to_numpy()
time_dummies_array = time_dummies.to_numpy()

# Concatenate the arrays
W3_combined = np.concatenate([W3, entity_dummies_array, time_dummies_array], axis=1)

# Run double LASSO regression to estimate alpha for D
estimated_beta_1, estimated_std_error = double_lasso(Y, D, W3_combined)
print("Estimated beta_1:", estimated_beta_1.round(4))
print("Estimated standard error:", estimated_std_error.round(4))
min = estimated_beta_1 - 1.96 * estimated_std_error
max = estimated_beta_1 + 1.96 * estimated_std_error
print("Confidence interval:", (min.round(4), max.round(4)))

# %%
# Run double LASSO regression to estimate alpha for D*F
estimated_beta_2, estimated_std_error = double_lasso(Y, DF, W2_combined)
print("Estimated beta_2:", estimated_beta_2.round(4))
print("Estimated standard error:", estimated_std_error.round(4))
min = estimated_beta_2 - 1.96 * estimated_std_error
max = estimated_beta_2 + 1.96 * estimated_std_error
print("Confidence interval:", (min.round(4), max.round(4)))
