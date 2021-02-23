import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def loss_fn(targets, outputs):
        loss =
        return
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outpus = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outpus = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)



class Model(nn.Module):
    def __init__(self, number_of_class=12, dropout,k_c=3):
        # k_c: kernel size of channel
        super(Net, self).__init__()
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

        #print(self)

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
        #return F.log_softmax(output, dim=1)
        return output
