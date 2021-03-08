import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from helps import edl_mse_loss
from sklearn import metrics


class NinaProDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.features[item, :], dtype=torch.float),
            "y": torch.tensor(self.targets[item, :], dtype=torch.float)
        }


class EngineTrain:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def criterion(outputs, targets, loss_params):  # loss function
        if loss_params['edl']:
            loss = edl_mse_loss(outputs, targets, loss_params)
        else:
            loss_fun = nn.CrossEntropyLoss()
            loss = loss_fun(outputs, targets)
        return loss

    def train(self, data_loaders, loss_params):
        final_loss = {}
        for phase in ['train', 'val']:
            train_flag = phase == 'train'
            self.model.train() if train_flag else self.model.eval()
            final_loss[phase] = 0.0
            data_n = 0.0
            for _, (inputs, targets) in enumerate(data_loaders[phase]):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)  # (batch_size,)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(train_flag):
                    outputs = self.model(inputs)  # (batch_size,class_n)
                    loss = self.criterion(outputs, targets, loss_params)
                    if train_flag:
                        loss.backward()
                        self.optimizer.step()
                final_loss[phase] += loss.item()*inputs.size(0)
                data_n += inputs.size(0)
            final_loss[phase] = final_loss[phase] / data_n
        return final_loss

    def re_train(self, data_loader, loss_params):
        final_loss = 0.0
        self.model.train()
        data_n = 0.0
        for _, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, loss_params)
                loss.backward()
                self.optimizer.step()

            final_loss += loss.item()*inputs.size(0)
            data_n += inputs.size(0)
            final_loss = final_loss / data_n
        return final_loss


class EngineTest:
    def __init__(self, outputs, targets):
        # outputs: numpy array; targets: numpy array
        self.outputs = outputs
        self.targets = targets

    @staticmethod
    def cal_recall(outputs, targets):
        _, true_class_n = np.unique(targets, return_counts=True)
        preds = outputs.argmax(dim=1, keepdim=True)
        recall = []
        for class_index, class_n in enumerate(true_class_n):
            true_each_class = targets == class_index
            pred_result_each_class = np.logical_and(preds, true_each_class)
            recall.append(np.sum(pred_result_each_class)/class_n)
        return recall

    def update_result_acc(self, params):
        # pred: prediction Results (not labels)
        # true: Ground truth labels
        # params: dict -->
        # {'sb_n': , edl', 'test_trial_list'}

        # load current result file
        filename = 'results/cv/accuracy.csv'
        column_names = [*params, 'gesture', 'recall']
        # 'sb_n','edl','test_trial_list'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=column_names)
        # Update it
        temp_dict = params
        recall = self.cal_recall(self.outputs, self.targets)
        for class_index, each_recall in enumerate(recall):
            temp_dict['gesture'] = i + 1
            temp_dict['recall'] = each_recall
            df = df.append([temp_dict])
        # Save it
        df.to_csv(filename, index=False)
        return df

    @staticmethod
    def cal_minAP(n_pos, n_neg):
        # theoretical minimum average precision
        AP_min = 0.0
        for i in range(1, n_pos+1):
            AP_min += i/(i+n_neg)
        AP_min = AP_min/n_pos
        return AP_min

    def cal_nAP(labels, scores):
        # labels: labels for postive or not
        # scores: quantified uncertainty
        n_sample = len(labels)  # the total number of predictions
        n_pos = np.sum(labels)  # the total number of positives
        n_neg = n_sample - n_pos  # the total number of negatives
        skew = n_pos/n_sample
        minAP = cal_minAP(n_pos, n_neg)

        AP = metrics.average_precision_score(labels, scores)
        nAP = (AP - minAP)/(1-minAP)  # normalised AP
        return nAP, skew

    def update_result_R(self, params):
        # pred: prediction Results (not labels)
        # true: Ground truth labels
        # params: dict -->
        # {'sb_n': , edl', 'test_trial_list'}
        # load current result file
        filename = 'results/cv/reliability.csv'
        column_names = [*params, 'gesture', 'skew', 'uncertainty', 'nAP']
        # *params: 'sb_n','edl','test_trial_list'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=column_names)
        # Update it

        _, true_class_n = np.unique(targets, return_counts=True)
        preds = outputs.argmax(dim=1, keepdim=True)

        for class_index, class_n in enumerate(true_class_n):
            true_each_class = targets == class_index
            pred_result_each_class = np.logical_and(preds, true_each_class)
            labels = np.logical_not(pred_result_each_class)

        return recall

    # Update acc
    df_acc = update_result_acc(df_acc, temp_data_acc, pred_results, targets)

    # Update mis

    # Get label  pos: wrong predictions; neg: right predictions
    labels = np.logical_not(pred_results)

        temp_dict = params


        for uncertainty_type, score in scores.items():
            temp = PREval(labels,score)
            temp_data['AP_nor'] = temp.AP_nor_cal()
            temp_data['uncertainty_type'] = uncertainty_type
            df = df.append([temp_data])


        recall = self.cal_recall(self.outputs, self.targets)
        for class_index, each_recall in enumerate(recall):
            temp_dict['gesture'] = i + 1
            temp_dict['recall'] = each_recall
            df = df.append([temp_dict])
        ## Save it
        df.to_csv(filename,index = False)
        return df


class Model(nn.Module):
    def __init__(self, number_of_class=12, dropout=0.5, k_c=3):
        # k_c: kernel size of channel
        super(Model, self).__init__()
        # self._batch_norm0 = nn.BatchNorm2d(1)
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(k_c, 5))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(dropout)
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 3))

        self._conv2 = nn.Conv2d(32, 64, kernel_size=(k_c, 5))
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(dropout)
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 3))

        self._fc1 = nn.Linear((16-2*k_c+2)*3*64, 500)
        # 8 = 12 channels - 2 -2 ;  53 = ((500-4)/3-4)/3
        self._batch_norm3 = nn.BatchNorm1d(500)
        self._prelu3 = nn.PReLU(500)
        self._dropout3 = nn.Dropout(dropout)

        self._output = nn.Linear(500, number_of_class)
        self.initialize_weights()

        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        # x = x.permute(0,1,3,2) # --> batch * 1 * 16 * 50
        # print(x.size())
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        # conv1 = self._dropout1(
        # self._prelu1(self._batch_norm1(self._conv1(self._batch_norm0(x)))))
        pool1 = self._pool1(conv1)
        conv2 = self._dropout2(
            self._prelu2(self._batch_norm2(self._conv2(pool1))))
        pool2 = self._pool2(conv2)
        flatten_tensor = pool2.view(pool2.size(0), -1)
        fc1 = self._dropout3(
            self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        output = self._output(fc1)
        return output
