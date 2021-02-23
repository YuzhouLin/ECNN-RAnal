import torch
import pandas as pd
import utilis
import pickle
from helps import load_data_cnn, get_device

DEVICE = get_device()
EPOCHS = 1
TRIAL_LIST = list(range(1,7))


def run_training(fold,params,hyperparams,save_model):
    # load_data
    temp_trial_list = [x for x in TRIAL_LIST if x not in params['test_trial_list']]
    valid_trial_list = [temp_trial_list.pop(fold)]
    train_trial_list = temp_trial_list

    # Get the train data

    train_loader = load_data_cnn(sb_n,trian_trial_list,hyperparams['batch_size'])
    valid_loader = load_data_cnn(sb_n,valid_trial_list,hyperparams['batch_size'])

    model = utilis.Model()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True)

    eng = utils.Engine(model,optimizer,device = DEVICE)

    best_loss = np.inf
    early_stopping_counter = 0
    for epoch in range(1,EPOCHS+1):
        train_loss = eng.train(train_loader,loss_params)
        valid_loss = eng.evaluate(valid_loader,loss_params)
        print(f"{fold}, {epoch}, {train_loss}, {valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), f"model_{fold}.bin")


if __name__ == "__main__":
    sb_n = 1
    test_trial_list = [5]
    inner_nested_cv_list =
    params = {
        'sb_n': sb_n,
        'test_trial_list': test_trial_list,
        'class_n': 12,
        'edl': False
    }
    hyperparams = {
        'batch_size': 128
    }
    run_training(1,params,hyperparams,save_model=False)
