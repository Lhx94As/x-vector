import os
import sys
import glob
import time
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

from data_load import MFCC_24
from x_vector_recipe import xvecTDNN

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

batch_size = 2048
num_epochs = 10  #more
learning_rate = 0.001

#== train model ==
# use_gpu = torch.cuda.is_available()
model = xvecTDNN(feature_dim=24,num_lang=10,p_dropout=0.2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
multi_gpu = True
if multi_gpu:
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model.to(device)

train_txt = '/home/hexin/Desktop/hexin/dataset/OLR2020/chunk/2017train/training_data.txt'
test_txt = '/home/hexin/Desktop/hexin/dataset/OLR2020/chunk/2017test/testing_data.txt'
train_set = MFCC_24(train_txt)
test_set = MFCC_24(test_txt)

train_data = DataLoader(
    dataset = train_set,
    batch_size = batch_size,
    shuffle = True)
test_data = DataLoader(
    dataset = test_set,
    batch_size = batch_size,
    shuffle = False,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = nn.DataParallel(optimizer, device_ids=[0, 1, 2, 3])


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Train the model
total_step = len(train_data)
curr_lr = learning_rate
for epoch in tqdm(range(num_epochs)):
    for step, (utt, labels) in enumerate(train_data):
        utt_ = utt.to(device)
        labels = labels.long().to(device)

        # Forward pass
        outputs = model(utt_) # output <=> prerdict_train
        loss = criterion(outputs, labels)
        # metric = metric_func(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))
            # print("Metric:{.4f}".format(metric.item()/step))
print(model.state_dict().keys())
torch.save(model.state_dict(), 'model.ckpt')