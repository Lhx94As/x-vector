import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from x_vector_recipe import *
from data_load import MFCC_data
from torch.utils.data import DataLoader
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--dim', type=int, help='dim of input features',
                        default=24)
    parser.add_argument('--modeldir', type=str, help='pre-trained model dir',
                        default='/home/hexin/Desktop/hexin/LID_project/LID_torch/model.ckpt')
    parser.add_argument('--data', type=str, help='data to be extracted x-vectors, in .txt')
    parser.add_argument('--xvdir', type=str, help='dir to save x-vectors',
                        default='/home/hexin/Desktop/hexin/LID_project/LID_torch/')
    parser.add_argument('--dataname',type=str, help='data name for save')
    args = parser.parse_args()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = MFCC_data(args.data)
    data = DataLoader(
        dataset=dataset,
        batch_size=1024,
        shuffle=False)
    total_step = len(data)
    model = xvec_extractor(feature_dim=args.dim)
    pretrained_dict = torch.load(args.modeldir)
    # print(pretrained_dict)
    model_dict = model.state_dict()
    pretrained_dict={ k : v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # extract x-vectors for data
    model.to(device)
    model.eval()
    embeddings = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(data):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            print('{} in {}'.format(step+1, total_step))
            if step == 0:
                embeddings = outputs
            else:
                embeddings = np.vstack((embeddings,outputs))
            # print(outputs.shape)
        np.save(args.xvdir+'/{}_x_vectors.npy'.format(args.dataname), embeddings)
if __name__ == "__main__":
    main()
