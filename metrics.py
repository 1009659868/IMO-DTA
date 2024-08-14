import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc,precision_recall_curve

#cindex
def get_cindex(Y, P):
    P = P[:,np.newaxis] - P
    P = np.float32(P==0) * 0.5 + np.float32(P>0)

    Y = Y[:,np.newaxis] - Y
    Y = np.tril(np.float32(Y>0), 0)

    P_sum = np.sum(P*Y)
    Y_sum = np.sum(Y)

    if Y_sum==0:
        return 0
    else:
        return P_sum/Y_sum
#Coefficient of Determination
def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

#Proportionality Constant
def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

#Adjusted Squared Error
def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

#Rm2 **which is a modification of R²**
def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

#R-squared **the method same as the "r_squared_error"**
def get_r2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    return r2

#Area Under the Precision-Recall Curve
def get_aupr(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true,y_pred)
    roc_aupr = auc(recall,precision)
    return roc_aupr

#Calculate Various Metrics
def calculate_metrics(pre_affinities, affinities):
    pre_affinities_tensor = torch.Tensor(pre_affinities)
    affinities_tensor = torch.Tensor(affinities)
    
    # MSE
    mse_loss_func = nn.MSELoss()
    mse_loss = mse_loss_func(pre_affinities_tensor, affinities_tensor)
    mse = mse_loss.item()
    
    # MAE
    mae_loss_func = nn.L1Loss()
    mae_loss = mae_loss_func(pre_affinities_tensor, affinities_tensor)
    mae = mae_loss.item()
    
    # RMSE
    # rmse = torch.sqrt(mse_loss).item()
    
    # R^2m
    rm2 = get_rm2(affinities, pre_affinities)

    #R^2
    r2 = get_r2(affinities, pre_affinities)

    # C-Index
    cindex = get_cindex(affinities, pre_affinities)


    return mse, mae, rm2, r2, cindex