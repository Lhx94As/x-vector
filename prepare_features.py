import librosa
import numpy as np
import glob
from tqdm import tqdm
import os
import torch.utils.data as data
import torch
import python_speech_features as psf
from sklearn.preprocessing import LabelEncoder
import argparse



def extract_MFCC(audiopath,order_1 = True,order_2 = True,mfccdim = 13):
    audioarray,sr_ = librosa.load(path=audiopath,sr=None)
    preemphasis = 0.97
    delta1 = []
    delta2 = []
    preemphasized = np.append(audioarray[0], audioarray[1:] - preemphasis * audioarray[:-1])
    mfcc = librosa.feature.mfcc(preemphasized, sr=sr_,n_mfcc=mfccdim,
                                hop_length=int(sr_/100), n_fft=int(sr_/40))  # if log-mel-filter-bank-energu
    # if order_1:
    #     delta1 = librosa.feature.delta(mfcc, order=1)
    #     mfcc_features = np.concatenate((mfcc, delta1))
    if order_2 and order_1:
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_features = np.concatenate((mfcc, delta1, delta2))
    else:
        mfcc_features = mfcc
    # raw_feature = mfcc(signal = audio,nfilt = 40) #if MFCC features are used, change corresponding codes
    # mfcc_features = np.concatenate((mfcc, delta1, delta2))
    # save MFCC features to local
    #save_path = r'E:\Dataset\MFCC'
    #if not os.path.exists(save_path):
        #os.mkdir(save_path)
    #audio_name = r'E:\Dataset\MFCC\mfcc_{0}.npy'.format(audiopath.split('/')[-1].split('.')[0])
    #np.save(audio_name,mfcc_features)
    # print ('log-filter-bank-energy is:\n'.format(raw_log_fbankenergy)) #print it if you would like to check
    return mfcc_features

def make_chunks(utt2lang,savepath,delta=True,acc=True,dim=13,chunk_size=100):
    with open(utt2lang, 'r') as f:
        reco_label_list = f.readlines()
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # need more ops for data and labels, since the reco_list may not be extract audio paths
    reco_list = [x.split(' ')[0] for x in reco_label_list]
    label_list = [x.split(' ')[1].replace('\n','') for x in reco_label_list]
    for reco_id in range(len(reco_list)):
        label = label_list[reco_id]
        reco_name = os.path.split(reco_list[reco_id])[-1].split('.')[0]
        MFCCs = extract_MFCC(audiopath=reco_list[reco_id],order_1=delta,order_2=acc,mfccdim=dim)
        print("Current MFCCs' shape:",MFCCs.shape)
        num_frames = MFCCs.shape[1]
        num_chunk = num_frames//chunk_size
        end = 0
        for chunk_id in range(num_chunk):
            start = chunk_id*chunk_size
            end = start+chunk_size
            MFCC_chunk = MFCCs[:,start:end]
            chunk_name = savepath +'/{}_{}_{}.npy'.format(label,reco_name,chunk_id+1)
            np.save(chunk_name,MFCC_chunk)
        # complement final chunk
        num_required_frames = chunk_size - (num_frames%chunk_size)
        start = end - num_required_frames
        final_chunk = MFCCs[:, start:]
        chunk_name = savepath + '/{}_{}_final.npy'.format(label, reco_name)
        np.save(chunk_name, final_chunk)

def make_feats(train, test, save_train, save_test, delta, acc, dim, chunk_size):
    '''

    :param train: utt2lang file for training data
    :param test: utt2lang file for testing data
    :param save_train: path to save training chunks
    :param save_test: path to save testing chunks
    :param delta: True if use delta MFCC
    :param acc: True if use acceleration MFCC
    :param dim: dim of MFCCs of each frame
    :param chunk_size: the input of NN should be of shape: (Batch, feat_dim, chunk_size)
    '''
    make_chunks(train, save_train, delta, acc, dim, chunk_size)
    make_chunks(test, save_test, delta, acc, dim, chunk_size)
    data_train = glob.glob(save_train+'/*npy')
    data_test = glob.glob(save_test+'/*npy')
    label_train = [os.path.split(x)[-1].split('_')[0] for x in data_train]
    label_test = [os.path.split(x)[-1].split('_')[0] for x in data_test]
    le = LabelEncoder()
    le.fit(label_train)
    label_train = le.transform(label_train)
    label_test = le.transform(label_test)
    with open(save_train+'/training_data.txt', 'w') as ff:
        for chunk_id in range(len(data_train)):
            info = '{} {}'.format(data_train[chunk_id],label_train[chunk_id])
            ff.write(info+'\n')
    with open(save_test+'/testing_data.txt', 'w') as ff:
        for chunk_id in range(len(data_test)):
            info = '{} {}'.format(data_test[chunk_id],label_test[chunk_id])
            ff.write(info+'\n')


def make_utt2lang(datapath,savepath):
    filelist = glob.glob(datapath+'/*wav')
    lan_list = [os.path.split(x)[-1].split('-')[0].split('_')[0] for x in filelist]
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    with open(savepath+'/utt2language.txt', 'w') as f:
        for i in tqdm(range(len(filelist))):
            info = '{} {}\n'.format(filelist[i], lan_list[i])
            f.write(info)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--chunk', type=int, help='size of chunk (number of frame)',
                        default=100)
    parser.add_argument('--train', type=str,help='utt2lang file for training data')
    parser.add_argument('--test', type=str, help='utt2lang file for testing data')
    parser.add_argument('--savetrain', type=str, help='path to save training chunks')
    parser.add_argument('--savetest', type=str, help='path to save testing chunks')
    parser.add_argument('--delta', type=bool, help='True if use delta MFCC',
                        default=True)
    parser.add_argument('--acc', type=bool, help='True if use acceleration MFCC',
                        default=True)
    parser.add_argument('--dim', type=int, help='dim of MFCCs',
                        default=13)
    args = parser.parse_args()
    ## =====change labels in offical utt2lang===========
    # datapath = 'F:/OLR2020/AP19-OLR_data/AP18-OLR_data/AP17-OLR_train_dev/data/train/audio'
    # savepath = 'F:/OLR2020/AP19-OLR_data/AP18-OLR_data/AP17-OLR_train_dev/data/train'
    # utt2lang_test = 'F:/OLR2020/AP19-OLR_data/AP18-OLR_data/utt2lang'
    # head = 'F:/OLR2020/AP19-OLR_data/AP18-OLR_data/AP17-OLR_test/data/test_all/audio'
    # with open(utt2lang_test,'r') as f:
    #     list = f.readlines()
    # with open(utt2lang_test+'uage.txt', 'w') as f:
    #     for i in list:
    #         new = '{} {}\n'.format(head+'/'+i.split()[0]+'.wav',i.split()[-1].split('-')[0])
    #         f.write(new)

    # print(list[0])
    # print(list[0].split()[-1].split('-')[0])
    # utt2lang_train = 'F:/OLR2020/AP19-OLR_data/AP18-OLR_data/AP17-OLR_train_dev/data/train/utt2language.txt'
    # with open(utt2lang_train,'r') as f:
    #     list = f.readlines()
    # print(list[0])
    # print(list[0].split())

    # make_utt2lang(datapath,savepath)
    # # ============= make chunk stuff ======================
    chunksize = 100
    # train ='F:/OLR2020/AP19-OLR_data/AP18-OLR_data/AP17-OLR_train_dev/data/train/utt2language.txt'
    # test = 'F:/OLR2020/AP19-OLR_data/AP18-OLR_data/AP17-OLR_test/data/utt2language.txt'
    savetrain = 'F:/OLR2020/chunk/2017train/training_data.txt'
    savetest = 'F:/OLR2020/chunk/2017test/testing_data.txt'
    with open(savetrain,'r') as f:
        list_train = f.readlines()
    with open(savetest,'r') as f:
        list_test = f.readlines()
    with open('F:/OLR2020/chunk/train.txt', 'w') as f:
        for i in list_train:
            new = i.replace('F:/OLR2020/chunk/2017train','/raid1/p3/xinyi/hexintemp/chunk/2017train')
            f.write(new)
    with open('F:/OLR2020/chunk/test.txt', 'w') as f:
        for i in list_test:
            new = i.replace('F:/OLR2020/chunk/2017test','/raid1/p3/xinyi/hexintemp/chunk/2017test')
            f.write(new)
    # make_feats(train,test,savetrain,savetest,delta=False,acc=False,dim=24,chunk_size=100)
