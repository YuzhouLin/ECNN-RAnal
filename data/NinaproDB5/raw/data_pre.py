import json
import pickle
DATA_PATH = '/cluster/home/cug/yl339/CNN_rawdata_TF_GPU/NinaPro/DB'

def update_data_config():
    temp_dict = {}
    result = []
    submit = './data_pre.json'

    ## Edit here
    # target data
    temp_dict['data_num'] = 5
    # target channel list
    temp_dict['channel_list'] = list(range(16))
    # target class list
    temp_dict['class_list'] = list(range(1,13))
    # target exercise
    temp_dict['ex_num'] = 1
    # target subject list
    temp_dict['sb_list'] = list(range(1,11))
    # target trial list
    temp_dict['trial_list'] = list(range(1,7))
    # sliding window length: wl (Unit: samples)
    # example:
    # if taking 250ms as a window and sampling frequency of data is 200Hz,
    # wl = (200Hz/1000ms) * (250ms) = 50 samples
    temp_dict['wl'] = 50
    # ratio_non_overlap [0-1]
    # if ratio_non_overlap = 0.1, size_non_overlap = wl*0.1
    temp_dict['ratio_non_overlap'] = 0.1
    result.append(temp_dict)

    with open(submit, 'w') as f:
        json.dump(result,f)
    return

def prepare_data(sb_list,class_list,channel_list,trial_list, wl, ratio_non_overlap):
    emg = io.loadmat(filepath)['emg']
    label = io.loadmat(filepath)['restimulus']
    cycle = io.loadmat(filepath)['rerepetition'] # cycle(trial)
    for sb_n in sb_list:
        data_path = DATA_PATH + str(data_n) + '/s' + str(sb_n)+'/S'+str(sb_n) + '_E' + str(ex_n) + '_A1.mat'
        for k in trial_list:
            X = []
            Y = []
            temp_file = 'sb'+str(sb_n)+'_trial'+str(k)+'.pkl'
            for m in class_list:
                samples = emg[np.nonzero(np.logical_and(cycle==k, label==m))[0]][:,channel_list]
                temp = []
                for n in range(0,samples.shape[0]-wl, int(wl*ratio_non_overlap)):
                    segdata = samples[n:n+self.window_samples,:] # 50*8
                    temp.append(np.expand_dims(segdata, axis=0))
                X.extend(temp)
                Y.extend(m-1+np.zeros(len(temp)))
            temp_dict = {'x': X, 'y':Y}
            f = open(temp_file,"wb")
            pickle.dump(temp_dict,f)
            f.close()
    return
if __name__ == "__main__":

    # Update the data config json file
    update_data_config()
    # load data config params from the config json file
    file = open('read_data_config.json','r')
    data_params = json.load(file)
    data_n = data_params['data_num'] # target dataset
    ex_n = data_params['ex_num'] # target exercise
    sb_list = data_params['sb_list'] # target subject list
    class_list = data_params['class_list'] # target class list
    channel_list = data_params['channel_list']# target channel list
    trial_list = data_params['trial_list']# target trial list
    # sliding window length: wl (Unit: samples)
    # example:
    # if taking 250ms as a window and sampling frequency of data is 200Hz,
    # wl = (200Hz/1000ms) * (250ms) = 50 samples
    wl = data_params['wl']
    # ratio_non_overlap [0-1]
    # if ratio_non_overlap = 0.1, size_non_overlap = wl*0.1
    ratio_non_overlap = data_params['ratio_non_overlap']

    # Save the data used for deep learning
    prepare(sb_list, class_list,channel_list,trial_list, wl, ratio_non_overlap)
