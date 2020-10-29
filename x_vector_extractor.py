import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from x_vector_recipe import *
from data_load import MFCC_data
from torch.utils.data import DataLoader
import numpy as np
import os
from collections import OrderedDict
def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--dim', type=int, help='dim of input features',
                        default=20)
    parser.add_argument('--modeldir', type=str, help='pre-trained model dir',
                        default='/home/hexin/Desktop/hexin/LID_project/LID_torch/model.ckpt')
    parser.add_argument('--data', type=str, help='data to be extracted x-vectors, in .txt')
    parser.add_argument('--xvdir', type=str, help='dir to save x-vectors',
                        default='/home/hexin/Desktop/hexin/LID_project/LID_torch/')
    parser.add_argument('--dataname',type=str, help='data name for save')
    args = parser.parse_args()


    if not os.path.exists(args.xvdir):
        os.mkdir(args.xvdir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = MFCC_data(args.data)
    data = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False)
    total_step = len(data)
    model = xvec_extractor(feature_dim=args.dim)
    pretrained_dict = torch.load(args.modeldir)
    new_state_dict = OrderedDict()
    # print(pretrained_dict)
    model_dict = model.state_dict()
    dict_list = []
    for k, v in model_dict.items():
        dict_list.append(k)
    # print(model_dict)
    # print(model_dict)
    for k, v in pretrained_dict.items():
        if k.startswith('module.') and k[7:] in dict_list:
            new_state_dict[k[7:]] = v
        elif k in dict_list:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    # extract x-vectors for data
    model.to(device)
    model.eval()
    embeddings = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(data):
            images = images.to(device)
            outputs = model(images)
            print('{} in {}'.format(step+1, total_step))
            if step == 0:
                embeddings = outputs
            else:
                embeddings = torch.cat((embeddings,outputs),dim=0)
            # print(embeddings.shape)
        embeddings = embeddings.cpu().numpy()
        np.save(args.xvdir+'/{}_x_vectors.npy'.format(args.dataname), embeddings)
if __name__ == "__main__":
    main()
