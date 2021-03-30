import argparse
import torch
import utils
import helps_pre as pre
import optuna


parser = argparse.ArgumentParser()
parser.add_argument(
    '--edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl')
args = parser.parse_args()


EDL_USED = args.edl
DEVICE = pre.get_device()
EPOCHS = 100
TRIAL_LIST = list(range(1, 7))
DATA_PATH = '/data/NinaproDB5/raw/'


def retrain(params):
    #  load_data
    train_trial_list = \
        [x for x in TRIAL_LIST if x != params['outer_f']]

    sb_n = params['sb_n']

    train_loader = pre.load_data_cnn(
        DATA_PATH, sb_n, train_trial_list, params['batch_size'])

    model = utils.Model()
    model.to(DEVICE)
    optimizer = getattr(
        torch.optim,
        params['optimizer_name'])(model.parameters(), lr=params['lr'])
    eng = utils.EngineTrain(model, optimizer, device=DEVICE)

    loss_params = {'edl_used': EDL_USED}

    if EDL_USED in [1, 2]:
        edl_loss_params = \
            ['kl', 'annealing_step', 'edl_fun', 'evi_fun', 'class_n']
        loss_params.update(
            {item: params.get(item) for item in edl_loss_params})
        loss_params['device'] = DEVICE

    best_loss = params['best_loss']
    for epoch in range(1, EPOCHS + 1):
        if EDL_USED:
            loss_params['epoch_num'] = epoch
        train_loss = eng.re_train(train_loader, loss_params)
        print(
            f"epoch:{epoch}, train_loss:{train_loss}",
            f"best_loss_from_cv:{best_loss}")
        if train_loss < best_loss:
            break

    torch.save(model.state_dict(), params['saved_model'])
    return


if __name__ == "__main__":

    params = {
        'class_n': 12,
        'batch_size': 128,
        'optimizer_name': 'Adam',
        'lr': 1e-3,
        'edl_used': EDL_USED
    }

    if EDL_USED in [1, 2]:
        params['edl_fun'] = 'mse'
        params['evi_fun'] = 'relu'
        params['annealing_step'] = 10
        params['kl'] = 0 if EDL_USED == 1 else 1

    # retraining and save the models
    for test_trial in range(1, 7):
        params['outer_f'] = test_trial
        for sb_n in range(1, 11):
            params['sb_n'] = sb_n
            core_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
            study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
            loaded_study = optuna.load_study(
                study_name="STUDY", storage=study_path)
            temp_best_trial = loaded_study.best_trial
            # Update for the optimal hyperparameters
            for key, value in temp_best_trial.params.items():
                params[key] = value

            model_name = f"models/ecnn{EDL_USED}/sb{sb_n}_t{test_trial}.pt"
            params['saved_model'] = model_name
            params['best_loss'] = temp_best_trial.value
            retrain(params)
