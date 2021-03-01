import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from helps import edl_mse_loss

class NinaProDataset:
    def __init__(self,features,targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.features[item,:],dtype = torch.float),
            "y": torch.tensor(self.targets[item,:],dtype = torch.float)
        }

class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def criterion(outputs, targets, loss_params):# loss function
        if loss_params['edl']:
            loss = edl_mse_loss(outputs,targets,loss_params)
        else:
            loss_fun = nn.CrossEntropyLoss()
            loss = loss_fun(outputs,targets)
        return loss
   
    @staticmethod
    def cal_recall(outputs, targets):
        _, true_class_n = np.unique(targets, return_counts=True)
        pred = outputs.argmax(dim=1, keepdim=True)
        recall = []
        for class_index, class_n in enumerate(true_class_n):
            true_each_class = targets == class_index
            pred_result_each_class = np.logical_and(preds, true_each_class)
            recall.append(np.sum(pred_result_each_class)/class_n)
        return recall


    def train(self, data_loaders, loss_params):
        final_loss = {}
        for phase in ['train','val']:
            train_flag = phase == 'train'
            self.model.train() if train_flag else self.model.eval()
            final_loss[phase] = 0.0
            data_n = 0.0
            for _, (inputs,targets) in enumerate(data_loaders[phase]):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device) # (batch_size,)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(train_flag):
                    outputs = self.model(inputs) # (batch_size,class_n) 
                    loss = self.criterion(outputs,targets,loss_params)
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
        for _, (inputs,targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs,targets,loss_params)
                loss.backward()
                self.optimizer.step()
            
            final_loss += loss.item()*inputs.size(0)
            data_n += inputs.size(0)
            final_loss = final_loss / data_n
        return final_loss

    def test(self, data_loader, edl_used): # full batch
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outputs = self.model(inputs)
            cal_recall(outputs, targets)
        return final_loss / len(data_loader)
    
class Model(nn.Module):
    def __init__(self, number_of_class=12, dropout=0.5, k_c=3):
        # k_c: kernel size of channel
        super(Model, self).__init__()
        #self._batch_norm0 = nn.BatchNorm2d(1)
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
        #x = x.permute(0,1,3,2) # --> batch * 1 * 16 * 50
        #print(x.size())
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        #conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(self._batch_norm0(x)))))
        pool1 = self._pool1(conv1)
        conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1))))
        pool2 = self._pool2(conv2)
        flatten_tensor = pool2.view(pool2.size(0),-1)
        fc1 = self._dropout3(self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        output = self._output(fc1)
        return output
