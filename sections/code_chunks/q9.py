# Create polynomial features of 5th order
poly = PolynomialFeatures(degree=5)
W_poly = poly.fit_transform(W_scaled_df)
W_p = pd.DataFrame(W_poly)
W_p.drop(columns=0, inplace=True)

# %%
# Create the design matrices
X9 = np.column_stack((D, PxD, W_p, entity_dummies, time_dummies))

# Fit Lasso regression models
alpha3 = BCCH(X9, Y)
lasso9 = Lasso(alpha=alpha3)  # You can adjust the alpha parameter as needed
lasso9.fit(X9, Y)

# Define the feature names
feature_names = [
    "D",
    "PD",
    "popestimate",
    "employment",
    "aveFareTotal",
    "VRHTotal",
    "VOMSTotal",
    "VRMTotal",
    "gasPrice",
]

# Create DataFrame for Model 9
coef9_df = pd.DataFrame({"Feature": feature_names, "Coefficient": lasso9.coef_[:9]})

print("Model 9 Coefficients:")
print(coef9_df)
