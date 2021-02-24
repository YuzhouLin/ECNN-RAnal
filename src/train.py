import argparse
import os
import torch
import numpy as np
#import torch.optim as optim
import pandas as pd
import utils
import pickle
from helps import load_data_cnn, get_device
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

parser = argparse.ArgumentParser()
parser.add_argument('-edl','--edl_used',help = 'if used edl?',action = 'store_true')
args = parser.parse_args()


DEVICE = get_device()
EPOCHS = 150
TRIAL_LIST = list(range(1,7))
DATA_PATH = '/data/NinaproDB5/raw/'

def run_training(fold,params,save_model):
    ## load_data
    temp_trial_list = [x for x in TRIAL_LIST if x not in params['test_trial_list']]
    valid_trial_list = [temp_trial_list.pop(fold)]
    train_trial_list = temp_trial_list

    sb_n = params['sb_n']

    train_loader = load_data_cnn(DATA_PATH,sb_n,train_trial_list,params['batch_size'])
    valid_loader = load_data_cnn(DATA_PATH,sb_n,valid_trial_list,params['batch_size'])
    
    trainloaders = {
    "train": train_loader,
    "val": valid_loader,
}


    model = utils.Model()
    model.to(DEVICE)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True)
    optimizer = getattr(torch.optim, params['optimizer_name'])(model.parameters(), lr=params['lr'])
    eng = utils.Engine(model,optimizer,device = DEVICE)

    edl_flag = params['edl']
    loss_params = {'edl':edl_flag}
    
    if edl_flag:
        edl_loss_params = ['kl','annealing_step','edl_fun','evi_fun','class_n']
        loss_params.update({item:params.get(item) for item in edl_loss_params})
        loss_params['device'] = DEVICE  
    
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    for epoch in range(1,EPOCHS+1):
        if edl_flag:
            loss_params['epoch_num'] = epoch
        train_losses = eng.train(trainloaders,loss_params) 
        train_loss = train_losses['train']
        valid_loss = train_losses['val']
        print(f"fold:{fold}, epoch:{epoch}, train_loss:{train_loss}, valid_loss:{valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), f"models/sb{sb_n}_{fold}.bin")
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            break
    
    return best_loss

def objective(trial,params):
    ## Update the params for tuning with cross validation
    params['optimizer_name'] = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    params['lr'] = trial.suggest_loguniform("lr", 1e-3, 1e-2)
    params['batch_size'] =  trial.suggest_int("batch_size", 128, 256, step = 128)
    if params['edl']:
        params['evi_fun'] =  trial.suggest_categorical("evi_fun", ["relu", "softplus", "exp"])
        params['kl'] = trial.suggest_int("KL",0,1,step=1)
        #params['annealing_step'] = trial.suggest_int("annealing_step",10,30,step=10)

    all_losses = []
    for f_ in range(5):
        temp_loss = run_training(f_,params,save_model=False)
        all_losses.append(temp_loss)

    return np.mean(all_losses)

if __name__ == "__main__":
    #sb_n = 1
    test_trial_list = [5]
    
    edl_used = args.edl_used

    params = {
        #'sb_n': sb_n,
        'test_trial_list': test_trial_list,
        'class_n': 12,
        'batch_size': 128,
        'optimizer_name': 'Adam',
        'lr': 1e-3,
        'edl': edl_used
    }
    if edl_used:
        params['edl_fun'] = 'mse'
        params['annealing_step'] = 10
        params['kl'] = 0
        params['evi_fun'] = 'relu'

    #run_training(1,params,save_model=False)

    for sb_n in range(1,11):
        params['sb_n'] = sb_n
        study_path = f'study/ecnn/sb{sb_n}' if edl_used else f'study/cnn/sb{sb_n}'
    
        if not os.path.exists(study_path):
            os.makedirs(study_path)

        sampler = TPESampler()
        study = optuna.create_study(
            direction="minimize",  # maximaze or minimaze our objective
            sampler=sampler,  # parametrs sampling strategy
            pruner=MedianPruner(
                n_startup_trials=15,
                n_warmup_steps=5,  # let's say num epochs
                interval_steps=2,
            ),
            study_name='STUDY',
            storage="sqlite:///"+study_path+"/temp.db",  # storing study results, other storages are available too, see documentation.
            load_if_exists=True
        )
        
        study.optimize(lambda trial: objective(trial,params), n_trials=25)
        
        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
