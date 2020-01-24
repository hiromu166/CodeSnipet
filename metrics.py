import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score, accuracy_score, recall_score, precision_score, confusion_matrix, log_loss
import seaborn as sns
import matplotlib.pyplot as plt

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def QWK(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def LogLoss(y_true, y_pred):
    return log_loss(y_true, y_pred)

def APR(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(acc)
    print(pre)
    print(rec)
    return acc, pre, rec

def Confusion_Matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues')

