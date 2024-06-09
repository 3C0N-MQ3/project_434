# Create the design matrices
X2 = pd.concat([D, FxD, W], axis=1)

# Fit the Panel OLS models with individual and time fixed effects
model4 = PanelOLS(Y, X2, entity_effects=True, time_effects=True, drop_absorbed=True)
result4 = model4.fit()

print(result4.summary)

# %%
#Create Residuals
ols3_res= result3.resids
ols4_res = result4.resids

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), constrained_layout=True)

# First plot
axes[0].scatter(ols3_res, np.log(data['UPTTotal']), label='Residuals', 
                color='b', alpha=0.6, edgecolor='w', s=80)
axes[0].set_xlabel('Residuals', fontsize=12)
axes[0].set_ylabel('Log of UPTTotal', fontsize=12)
axes[0].set_title('Residuals vs. UPTTotal for OLS with DP Interaction', fontsize=14)
axes[0].legend()
axes[0].grid(True, which='both', linestyle='--', linewidth=0.7)

# Second plot
axes[1].scatter(ols4_res, np.log(data['UPTTotal']), label='Residuals', 
                color='r', alpha=0.6, edgecolor='w', s=80)
axes[1].set_xlabel('Residuals', fontsize=12)
axes[1].set_ylabel('Log of UPTTotal', fontsize=12)
axes[1].set_title('Residuals vs. UPTTotal for OLS with DF Interaction', fontsize=14)
axes[1].legend()
axes[1].grid(True, which='both', linestyle='--', linewidth=0.7)

plt.show()
# %%
# Create Figure
obs_count = list(range(1, len(ols3_res) + 1))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), constrained_layout=True)

# First plot
axes[0].plot(obs_count, ols3_res, label='Residual', color='b', linewidth=2, 
             marker='o', markersize=5, markerfacecolor='white')
axes[0].set_xlabel('Observation', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('Residuals with DP Interaction', fontsize=14)
axes[0].legend()
axes[0].grid(True, which='both', linestyle='--', linewidth=0.7)

# Second plot
axes[1].plot(obs_count, ols4_res, label='Residual', color='r', linewidth=2, 
             marker='o', markersize=5, markerfacecolor='white')
axes[1].set_xlabel('Observation', fontsize=12)
axes[1].set_ylabel('Residual', fontsize=12)
axes[1].set_title('Residuals OLS with DF Interaction', fontsize=14)
axes[1].legend()
axes[1].grid(True, which='both', linestyle='--', linewidth=0.7)

plt.show()