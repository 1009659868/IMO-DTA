import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', ' ').split()
        data = list(map(float, data))  # 转换为浮点数
    return np.array(data)

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

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def loss_func(pre_affinities, affinities):
    return F.mse_loss(pre_affinities, affinities)

# 计算RMSE、MAE和MSE
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
    # C-Index
    cindex = get_cindex(affinities, pre_affinities)
    
    return mse, mae, rm2, cindex


if __name__ == "__main__":
    
    for i in range(5):
        print(f"No.{i}-----------------------------------------")
        # 读取数据
        affinity = f'./test/iter{i}affinities.txt'
        affinities = read_data(affinity)
        
        preaffinity = f'./test/iter{i}preaffinities.txt'
        preaffinities = read_data(preaffinity)
    
        print("affinities", affinities)
        print("preaffinities", preaffinities)
        
        mse, mae, rm2, cindex = calculate_metrics(preaffinities, affinities)
        
        
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"RM2: {rm2}")
        print(f"C-Index: {cindex}")

        print(f"No.{i}----------------------------------------------\n")
    
