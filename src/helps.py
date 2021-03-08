import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels, num_classes):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def load_data_cnn(data_path, sb_n, trial_list, batch_size):
    X = []  # L*1*16(channels)*50(samples)
    Y = []
    for trial_n in trial_list:
        temp = pd.read_pickle(
            os.getcwd() + data_path + f"sb{sb_n}_trial{trial_n}.pkl")
        X.extend(temp['x'])
        Y.extend(temp['y'])
    data = TensorDataset(
        torch.from_numpy(np.array(X, dtype=np.float32)).permute(0, 1, 3, 2),
        torch.from_numpy(np.array(Y, dtype=np.int64)))
    if batch_size > 1:  # For training and validation
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, drop_last=True)
    elif batch_size == 1:  # For testing
        # default DataLoader: batch_size = 1, shuffle = False, drop_last =False
        data_loader = torch.utils.data.DataLoader(data)
    return data_loader


def load_data_test_cnn(data_path, sb_n, trial_list):
    X = []  # L*1*16(channels)*50(samples)
    Y = []
    for trial_n in trial_list:
        temp = pd.read_pickle(
            os.getcwd() + data_path + f'sb{sb_n}_trial{trial_n}.pkl')
        X.extend(temp['x'])
        Y.extend(temp['y'])

        X = torch.as_tensor(
            torch.from_numpy(np.array(self.X))).permute(0, 1, 3, 2)
        # L*1*16*50
        # Y = torch.from_numpy(np.array(self.Y, dtype=np.int64))
        Y = np.array(self.Y, dtype=np.int64)
        # X: tensor; Y: numpy
        return X, Y


def relu_evidence(y):
    return F.relu(y)


def softmax_evidence(y):
    return F.softmax(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, max=3))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, params):
    # if annealing_step = 0, no kl
    y = y.to(params['device'])  # 256*12
    alpha = alpha.to(params['device'])  # 256*12
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = loglikelihood_loss(y, alpha, device=params['device'])
    '''
    belief = (alpha-1.0)/S
    batch_n = belief.size()[0]
    u_dis = torch.zeros(batch_n,1).to(device)
    for index_k in range(num_classes):
        temp0 = torch.zeros(batch_n,1).to(device)
        temp1 = torch.zeros(batch_n,1).to(device)
        for index_j in range(num_classes):
            if index_j!=index_k:
                k = belief[:,index_k].reshape(batch_n ,1).to(device)
                j = belief[:,index_j].reshape(batch_n ,1).to(device)
                temp0 += j*(1.0-torch.abs(k-j)/(k+j+1e-8))
                temp1 += j
        u_dis += k*temp0/(temp1+1e-8)
    '''
    annealing_coef = \
        torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(
                params['epoch_num'] / params['annealing_step'],
                dtype=torch.float32)
        )

    if params['kl'] == 0:
        # return loglikelihood_err + loglikelihood_var
        # return 0.6*(loglikelihood_err + loglikelihood_var) + 0.4*u_dis
        return loglikelihood
    else:
        kl_alpha = (alpha - 1) * (1 - y) + 1
        target_alpha = torch.sum(alpha * y, dim=1, keepdim=True)
        # p_t = target_alpha/S
        # print(target_alpha.size())
        # torch.sum(alpha[y==1],dim=1,keepdim=True)
        # u = num_classes/S

        # A = loglikelihood_err + loglikelihood_var
        # cond_coef = torch.where(loglikelihood_err>0.5,1.0,-1.0)

        # print(cond_coef)
        # loss = A - annealing_coef*(1.-p_t)**2*u

        # loss = A + (loglikelihood_err-0.5)**2*u

        # cond_coef*(1.-p_t)**2*u

        # return loss
        # total_S = torch.sum(alpha,dim=1,keepdim=True)
        # print(total_S)
        # u = u*annealing_coef
        kl_div = annealing_coef * \
            kl_divergence(kl_alpha, params['class_n'], device=params['device'])
        # a = torch.tensor(0.8, dtype=torch.float32)
        return loglikelihood + kl_div  # + (1-p_t)*kl_div


def edl_mse_loss(output, target, params):
    evidence = eval(params['evi_fun'] + '_evidence(output)')
    alpha = evidence + 1
    y = one_hot_embedding(target, params['class_n'])
    loss = torch.mean(mse_loss(y.float(), alpha, params))
    return loss
