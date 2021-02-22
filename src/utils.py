import torch
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
