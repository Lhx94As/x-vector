import sys
import os
import random
import glob
import librosa
from tqdm import tqdm

# this file use kaldi to do data augmentation and reverberation
#Updating in progress...

rirs_dir = '/home/hexin/Desktop/hexin/dataset/RIRS_NOISES/real_rirs_isotropic_noises'
wav_dir = '/home/hexin/Desktop/hexin/dataset/NRF_xv/Eng_train2'
save_dir = '/home/hexin/Desktop/hexin/dataset/NRF_xv/Eng_aug'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
data_aug = True
data_reverb = True
rirs_noise = glob.glob(rirs_dir+'/*wav')
snr_list = [5.0, 10.0, 15.0,20.0]
wav_list = glob.glob(wav_dir+'/*wav')
for fi in wav_list:
    data, sr = librosa.load(fi, sr=None)
    dur = len(data)/sr
    source_wav = fi
    print(source_wav)
    snr = random.choice(snr_list)
    #for noise in os.listdir(rirs_dir):
    rirs_wav = random.choice(rirs_noise)
    name = os.path.split(fi)[-1].split('.wav')[0]
    save_name = name+'_rirs.wav'
    save_wav = os.path.join(save_dir, save_name)
    #rirs_wav = random.choice(rirs_noise)
    #rirs_wav = os.path.join(rirs_dir, rirs_wav)
    #print("rirs_wav", rirs_wav)
    if data_aug:
        os.system("/home/hexin/Desktop/hexin/kaldi/src/featbin/wav-reverberate --shift-output=true "
                  "--impulse-response='sox {1} -r {5} -t wav - |' --start-times='0' "
                  "--duration={4} --snrs='{0}' {2} {3}".format(snr,rirs_wav, source_wav, save_wav, dur, sr))
    # if data_reverb:
    #     os.system("/home/hexin/Desktop/hexin/kaldi/src/featbin/wav-reverberate --shift-output=true "
    #               "--additive-signals='sox ../kaldi-master/simulated_rirs/white.wav -r 16000 -t wav - |' "
    #               "--start-times='0' --snrs='%f' %s %s" %(snr, source_wav, save_wav))
    print(save_name)
