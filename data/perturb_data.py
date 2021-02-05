import glob
import subprocess
import os
import argparse
import warnings
import random
from tqdm import tqdm
# This file include two ways to do data augmentation: speed perturbation and volume perturbation.
# The perturbation methods followed kaldi's recipe but I simplify them, you can also find them in kaldi's github.

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--data', type=str, help='data to be peturbed',
                        default='/home/hexin/Desktop/hexin/dataset/OLR2020/perd/train/audio')
    parser.add_argument('--spd', type=str, help='dir to save speed perturbed data',
                        default='/home/hexin/Desktop/hexin/dataset/OLR2020/perd/train/audio/')
    parser.add_argument('--vol', type=str, help='dir to save volume perturbed data',
                        default='/home/hexin/Desktop/hexin/dataset/OLR2020/perd/train/audio/')
    parser.add_argument('--spdfac',type=float, help='speed',default=0.9)
    parser.add_argument('--random_vol', type=bool,
                        help='if True, random volume factors are used for each seg, otherwise 0.125 and 2',
                        default=False)
    args = parser.parse_args()
    root = args.data
    filelist = glob.glob(root+'/*wav')
    factor1 = args.spdfac
    factor2 = 2.0-factor1
    scale_low = 0.125
    scale_high = 2
    if args.random_vol:
        factor_range = [0.2, 0.3, 0.4, 0.5, 0.6, 1.6, 1.7, 1.8, 1.9, 2.0]
        factor_list = [random.choice(factor_range) for i in range(len(filelist))]
    if not os.path.exists(args.vol):
        os.mkdir(args.vol)
    if not os.path.exists(args.spd):
        os.mkdir(args.spd)
    for i in tqdm(range(len(filelist))):
        file = filelist[i]
        new1 = args.spd+'/'+'sp-{}_'.format(factor1)+os.path.split(file)[-1]
        new2 = args.spd+'/'+'sp-{}_'.format(factor2)+os.path.split(file)[-1]
        subprocess.call(f"sox -t wav {file} -t wav {new1} speed {factor1}",shell=True)
        subprocess.call(f"sox -t wav {file} -t wav {new2} speed {factor2}", shell=True)
        if args.random_vol:
            vol_scale = factor_list[i]
            new_vol = args.vol + '/' + 'vol-{}_'.format(vol_scale) + os.path.split(file)[-1]
            subprocess.call(f"sox --volume {vol_scale} {file} {new_vol}", shell=True)
        else:
            new_low = args.vol + '/' + 'vol-{}_'.format(scale_low) + os.path.split(file)[-1]
            new_high = args.vol + '/' + 'vol-{}_'.format(scale_high) + os.path.split(file)[-1]
            subprocess.call(f"sox --volume {scale_high} {file} {new_high}", shell=True)
            subprocess.call(f"sox --volume {scale_low} {file} {new_low}", shell=True)

if __name__ == "__main__":
    main()
