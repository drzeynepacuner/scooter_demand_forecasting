import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    # fallback
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-15, None))) * 100

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)
