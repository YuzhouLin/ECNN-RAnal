import argparse
#import utils
import optuna
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    'edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl')
args = parser.parse_args()


EDL_USED = args.edl
CLASS_N = 12
TRIAL_LIST = list(range(1, 7))

if __name__ == "__main__":
    for sb_n in range(1, 11):
        print('sb: ', sb_n)
        for test_trial in range(1, 7):
            core_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
            study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
            loaded_study = optuna.load_study(
                study_name="STUDY", storage=study_path)
            temp_best_trial = loaded_study.best_trial
            print('test_trial: ', test_trial)
            for key, value in temp_best_trial.params.items():
                print(key, value)
