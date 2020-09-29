import numpy as np
import torch.utils.data as data
import torch

class MFCC_24(data.Dataset):
    def __init__(self,txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [i.split()[0] for i in lines]
            self.label_list = [float(i.split()[-1]) for i in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        label = self.label_list[index]
        feature = torch.from_numpy(np.load(feature_path,allow_pickle=True))
        return feature, label

    def __len__(self):
        return len(self.label_list)
