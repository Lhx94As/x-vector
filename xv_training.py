import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_load import MFCC_data
from x_vector_recipe import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    
def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--dim', type=int, help='dim of input features',
                        default=20)
    parser.add_argument('--modeldir', type=str, help='dir to save model',
                        default='Your default dir + model_name.ckpt')
    parser.add_argument('--data', type=str, help='training data, in .txt')
    parser.add_argument('--batch', type=int, help='batch size',
                        default=2048)
    parser.add_argument('--epochs',type=int, help='num of epochs',
                        default=20)
    parser.add_argument('--lang',type=int, help='num of language classes',
                        default=10)
    parser.add_argument('--lr',type=float,help='initial learning rate',
                        default=0.001)
    parser.add_argument('--multigpu', type=bool, help='True if use multiple GPUs to train',
                        default=True)
    args = parser.parse_args()

    setup_seed(0)
 
    #== train model ==
    model = xvecTDNN(feature_dim=args.dim,num_lang=10,p_dropout=0.2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    multi_gpu = args.multigpu
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

    train_txt = args.data
    train_set = MFCC_data(train_txt)
    train_data = DataLoader(
        dataset = train_set,
        batch_size = args.batch,
        pin_memory=True,
        num_workers=16,
        shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    
    # Train the model
    total_step = len(train_data)
    curr_lr = args.lr
    for epoch in tqdm(range(args.epochs)):
        for step, (utt, labels) in enumerate(train_data):
            utt_ = utt.to(device)
            labels = labels.long().to(device)

            # Forward pass
            outputs = model(utt_) # output <=> prerdict_train
            loss = criterion(outputs, labels)
            # metric = metric_func(outputs, labels) # e.g.: EER, accuracy

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))
                # print("Metric:{.4f}".format(metric.item()))
        scheduler.step()
    torch.save(model.state_dict(), args.modeldir)

if __name__ == "__main__":
    main()
