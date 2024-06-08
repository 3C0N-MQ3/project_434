import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .BCCH import BCCH

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