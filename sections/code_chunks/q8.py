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