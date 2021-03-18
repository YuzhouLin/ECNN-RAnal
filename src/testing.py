import argparse
import os
import torch
import numpy as np
import pandas as pd
import utils
import pickle
import helps_pre as pre
import optuna
import copy
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument(
    '-edl', '--edl_used', help='if used edl?', action='store_true')
args = parser.parse_args()


DEVICE = pre.get_device()
DATA_PATH = '/data/NinaproDB5/raw/'

def test(params):
    #  load_data
    device = torch.device('cpu')
    # test_trial = params['outer_f']
    # sb_n = params['sb_n']

    # Load testing Data
    inputs, targets = pre.load_data_test_cnn(
        DATA_PATH, params['sb_n'],
        params['outer_f'])

    # Load trained model
    model = utils.Model()
    model.load_state_dict(
        torch.load(params['saved_model'], map_location=device))
    model.eval()

    # Get Results
    outputs = model(inputs.to(device)).detach()

    # Load the Testing Engine
    eng = utils.EngineTest(outputs, targets)

    common_keys_for_update_results = ['sb_n', 'edl', 'outer_f']

    dict_for_update_acc = \
        {key: params[key] for key in common_keys_for_update_results}
    dict_for_update_R = copy.deepcopy(dict_for_update_acc)

    eng.update_result_acc(dict_for_update_acc)

    if params['edl']:
        dict_for_update_R['acti_fun'] = params['evi_fun']
    else:
        dict_for_update_R['acti_fun'] = 'softmax'

    print(dict_for_update_R)
    eng.update_result_R(dict_for_update_R)

    return


if __name__ == "__main__":
   
    edl_used = args.edl_used
    params = {
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

    # retraining and save the models 
    for test_trial in range(1, 7):
        params['outer_f'] = test_trial
        for sb_n in range(1, 11):
            params['sb_n'] = sb_n

            if edl_used:
                model_name = f"models/ecnn/sb{sb_n}_t{test_trial}.pt"
            else:
                model_name = f"models/cnn/sb{sb_n}_t{test_trial}.pt"

            params['saved_model'] = model_name

            test(params)
            print(f'Testing done. sb{sb_n}-t{test_trial}')

            '''
            acc = pd.read_csv('results/cv/accuracy.csv')
            print(acc)
            R = pd.read_csv('results/cv/reliability.csv')
            print(R)
            exit()
            '''
