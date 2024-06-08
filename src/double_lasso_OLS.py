import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .BCCH import BCCH

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