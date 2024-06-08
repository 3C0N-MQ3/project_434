import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def BCCH(X, Y, c=1.1, alpha=0.05):
    X_copy = np.array(X)
    Y_copy = np.array(Y)

    if len(X_copy) != len(Y_copy):
        raise ValueError("Length of X and Y must be the same")

    n, p = X_copy.shape

    Y_reshaped = np.tile(Y_copy, (p, 1)).T
    maximum = np.max((np.mean((X_copy * Y_reshaped) ** 2, axis=1)) ** 0.5)
    ppf = norm.ppf(1 - alpha / (2 * p))
    lambda_pilot = (c / (n ** 0.5)) * ppf * maximum
    lasso = Lasso(alpha=lambda_pilot)                                 
    lasso.fit(X_copy, Y_copy)
    Y_prediction = lasso.predict(X_copy).reshape(-1, 1)
    e = Y_copy - Y_prediction.flatten()
    e_reshaped = np.tile(e, (p, 1)).T
    new_maximum = np.max((np.mean((X_copy * e_reshaped) ** 2, axis=1)) ** 0.5)
    lambda_final = (c / (n ** 0.5)) * ppf * new_maximum
    return lambda_final