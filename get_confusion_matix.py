import numpy as np
import torch
import src.utils as utils
import src.helps_pre as pre
from sklearn.metrics import confusion_matrix
import optuna


DEVICE = pre.get_device()
DATA_PATH = '/data/NinaproDB5/raw/'


def test(params):
    #  load_data
    device = torch.device('cpu')
    test_trial = params['outer_f']
    sb_n = params['sb_n']

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
    results = eng.get_output_results(params['acti_fun'])
    preds = results.argmax(axis=1)
    return targets, preds

if __name__ == "__main__":

    # Save the CM_matrix seperately
    CM_model = []
    for edl_used in range(4):
        params = {'edl_used': edl_used}
        CM_cv=[]
        for test_trial in range(1, 7):
            params['outer_f'] = test_trial
            CM_sb=[]
            for sb_n in range(1, 11):
                # Get the optimal activation function
                if edl_used == 0:
                    params['acti_fun'] = 'softmax'
                else:
                    # Get from hyperparameter study
                    core_path = f'study/ecnn{edl_used}/sb{sb_n}'
                    study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
                    loaded_study = optuna.load_study(
                        study_name="STUDY", storage=study_path)
                    temp_best_trial = loaded_study.best_trial
                    params['acti_fun']  = temp_best_trial.params['evi_fun']
                params['sb_n'] = sb_n
                model_name = f"models/ecnn{edl_used}/sb{sb_n}_t{test_trial}.pt"
                params['saved_model'] = model_name
                y_true, y_pred = test(params)
                print(f'Testing done. sb{sb_n}-t{test_trial}')
                #ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
                CM_sb.append(confusion_matrix(y_true, y_pred))
            CM_cv.append(CM_sb)
        CM_model.append(CM_cv)
    np.save('results/cv/CM_each', CM_model)

    '''
    # Sum up the CM matrix
    CM_matrix = []
    for edl_used in range(4):
        params = {'edl_used': edl_used}
        total_true = np.empty([0,0])
        total_pred = np.empty([0,0])
        for test_trial in range(1, 7):
            params['outer_f'] = test_trial
            for sb_n in range(1, 11):
                # Get the optimal activation function
                if edl_used == 0:
                    params['acti_fun'] = 'softmax'
                else:
                    # Get from hyperparameter study
                    core_path = f'study/ecnn{edl_used}/sb{sb_n}'
                    study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
                    loaded_study = optuna.load_study(
                        study_name="STUDY", storage=study_path)
                    temp_best_trial = loaded_study.best_trial
                    params['acti_fun']  = temp_best_trial.params['evi_fun']
                params['sb_n'] = sb_n
                model_name = f"models/ecnn{edl_used}/sb{sb_n}_t{test_trial}.pt"
                params['saved_model'] = model_name
                y_true, y_pred = test(params)
                print(f'Testing done. sb{sb_n}-t{test_trial}')
                #ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
                total_true = y_true if len(total_true) == 0 else np.append(total_true, y_true)
                total_pred = y_pred if len(total_pred) == 0 else np.append(total_pred, y_pred)
        CM_matrix.append(confusion_matrix(total_true, total_pred))
    np.save('results/cv/CM_sum', CM_matrix)
    '''