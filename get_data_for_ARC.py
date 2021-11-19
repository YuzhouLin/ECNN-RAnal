import os
import numpy as np
import torch
import src.utils as utils
import src.helps_pre as pre
import src.helps_pro as pro
import optuna
import pandas as pd


DEVICE = pre.get_device()
DATA_PATH = '/data/NinaproDB5/raw/'

def update_threshold(valid_pred_total, pred_results_total):
    
    # n: the number of testing samples
    n = len(pred_results_total)
    # n_accept: the number of accepted predictions
    n_accept = np.sum(valid_pred_total)
    # n_reject: the number of rejections 
    n_reject = n-n_accept
    # rr: rejection rate
    rr = np.round(n_reject/n, decimals=4)

    # accept_pred: the accepted predictions
    accept_pred = pred_results_total[valid_pred_total]

    # n_pos_accept: the number of accepted right predictions
    n_pos_accept = np.sum(accept_pred)

    # tar: true positive acceptance rate
    tar = np.round(n_pos_accept/(n_accept+1e-08), decimals=4)

    return rr, tar

def update_data_for_ARC(params):
    #  load_data
    device = torch.device('cpu')
    test_trial = params['outer_f']
    sb_n = params['sb_n']

    temp_dict = {'edl_used': params['edl_used'], 'sb': sb_n, 'test_trial': test_trial}
    column_names = [*temp_dict, 'threshold', 'RR', 'TAR']
    df_temp = pd.DataFrame(columns=column_names)
    # Load testing Data
    inputs, targets = pre.load_data_test_cnn(
        DATA_PATH, sb_n, test_trial)

    # Load trained model
    model = utils.Model()
    model.load_state_dict(
        torch.load(params['saved_model'], map_location=device))
    model.eval()

    # Get Results
    outputs = model(inputs.to(device)).detach()

    # Load the Testing Engine
    eng = utils.EngineTest(outputs, targets)
    output_results = eng.get_output_results(params['acti_fun'])
    #  Calculate the scores
    scores = pro.cal_scores(output_results, params['edl_used'])
    
    un = scores['overall']
    threshold_index = np.arange(0, 1, 0.02)
    pred_results = eng.get_pred_results()

    for threshold in threshold_index:
        valid_pred = un<(1-threshold) # (1547,1)
        rr, tar = update_threshold(valid_pred, pred_results)
        temp_dict['threshold'] = threshold
        temp_dict['RR'] = rr
        temp_dict['TAR'] = tar
        df_temp = df_temp.append([temp_dict])

    return df_temp

if __name__ == "__main__":
    # save data to the file 
    filename = 'results/cv/data_for_ARC.csv'
    column_names = ['edl_used', 'sb', 'test_trial', 'threshold', 'RR', 'TAR']
    df = pd.DataFrame(columns=column_names)
    for edl_used in range(4):
        params = {'edl_used': edl_used}        
        for test_trial in range(1, 7):
            params['outer_f'] = test_trial
            for sb_n in range(1, 11):
                # Get the optimal activation function
                if edl_used == 0:
                    params['acti_fun'] = 'softmax'
                else:
                    # Get from hyperparameter study
                    core_path = f"study/ecnn{edl_used}/sb{sb_n}"
                    study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
                    loaded_study = optuna.load_study(
                        study_name="STUDY", storage=study_path)
                    temp_best_trial = loaded_study.best_trial
                    params['acti_fun']  = temp_best_trial.params['evi_fun']
                params['sb_n'] = sb_n
                model_name = f"models/ecnn{edl_used}/sb{sb_n}_t{test_trial}.pt"
                params['saved_model'] = model_name
                df_temp = update_data_for_ARC(params)
                df = df.append(df_temp) # to be c
                print('Updating done')
    df.to_csv(filename, index=False)
    print(df)