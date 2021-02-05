import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import argparse
from data_load import MFCC_data
from x_vector_recipe import xvecTDNN

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--dim', type=int, help='dim of input features',
                        default=20)
    parser.add_argument('--modeldir', type=str, help='dir to save model',
                        default='Your default dir + model_name.ckpt')
    parser.add_argument('--data', type=str, help='training data, in .txt')
    parser.add_argument('--batch', type=int, help='batch size',
                        default=2048)
    parser.add_argument('--epoch',type=int, help='num of epochs',
                        default=20)
    parser.add_argument('--lang',type=int, help='num of language classes',
                        default=10)
    parser.add_argument('--lr',type=float,help='initial learning rate',
                        default=0.001)
    parser.add_argument('--multigpu', type=bool, help='True if use multiple GPUs to train',
                        default=True)
    args = parser.parse_args()

    setup_seed(0)

    batch_size = args.batch
    num_epochs = args.epoch  #more
    learning_rate = args.lr

    #== train model ==
    # use_gpu = torch.cuda.is_available()
    model = xvecTDNN(feature_dim=args.dim,num_lang=10,p_dropout=0.2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    multi_gpu = args.multigpu
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

    train_txt = args.data
    # test_txt = '/home/hexin/Desktop/hexin/dataset/OLR2020/chunk/save_txt/testing_data.txt'
    train_set = MFCC_data(train_txt)
    # test_set = MFCC_data(test_txt)

    train_data = DataLoader(
        dataset = train_set,
        batch_size = batch_size,
        pin_memory=True,
        num_workers=16,
        shuffle = True)
    # test_data = DataLoader(
    #     dataset = test_set,
    #     batch_size = batch_size,
    #     shuffle = False,
    # )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = nn.DataParallel(optimizer, device_ids=[0, 1, 2, 3])

    # Train the model
    total_step = len(train_data)
    curr_lr = args.lr
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
        if (epoch + 1) % 3 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
    # print(model.state_dict().keys())
    torch.save(model.state_dict(), args.modeldir)

if __name__ == "__main__":
    main()
