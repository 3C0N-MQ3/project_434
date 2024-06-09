# Convert 'dateSurvey' to datetime format
data["dateSurvey"] = pd.to_datetime(data["dateSurvey"], errors="coerce")

# Set the index to be a MultiIndex for panel data
data = data.set_index(["agency_city", "dateSurvey"])

# Define the dependent variable and independent variables
Y = np.log(data["UPTTotal"])
D = data["treatUberX"]
W = data[
    [
        "popestimate",
        "employment",
        "aveFareTotal",
        "VRHTotal",
        "VOMSTotal",
        "VRMTotal",
        "gasPrice",
    ]
]
PxD = data["PxD"]
FxD = data["FxD"]

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
