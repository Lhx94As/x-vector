import sys
import os
import random
import glob
import librosa
from tqdm import tqdm

noises_dir = '' # Your noise path
wav_dir = '' # Your data to be processed
save_dir = '' # Your save dir
snr_list = [5, 10 ,15, 20]
snr_level = 20 # The target SNR level
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_aug = False # if True, add noise to speech
random_snr = False # if True, randomly choose SNR from the snr list, otherwise use snr_level
data_reverb = True # if True, do reverberation
wav_list = glob.glob(wav_dir+'/*wav')
noises = glob.glob(noises_dir+'/*wav')

for fi in wav_list:
    data, sr = librosa.load(fi, sr=None)
    dur = len(data)/sr
    source_wav = fi
    if random_snr:
        snr = random.choice(snr_list)
    else:
        snr = snr_level
    noise_wav = random.choice(noises)
    name = os.path.split(fi)[-1].split('.wav')[0]
    # data_aug: add noise to the speech; data reverb: reverberate the speech
    if data_aug:
        save_name = name + '_noisy.wav'
        save_wav = os.path.join(save_dir, save_name)
        os.system("{}/src/featbin/wav-reverberate --shift-output=true ".format(kaldi_root)
                  "--additive-signals='sox {1} -r 16000 -t wav - |' --start-times='0' "
                  "--snrs='{0}' {2} {3}".format(snr, noise_wav, source_wav, save_wav))

    if data_reverb:
        save_name = name + '_reverb.wav'
        save_wav = os.path.join(save_dir, save_name)
        os.system("{}/src/featbin/wav-reverberate --shift-output=true ".format(kaldi_root)
                  "--impulse-response='sox {0} -r 16000 -t wav - |' "
                  "--start-times='0' --snrs='0.0' {1} {2}".format(noise_wav, source_wav, save_wav))
