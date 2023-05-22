from collections import namedtuple
import os
import math
import random
from tkinter import W
import pandas as pd
import numpy as np
from tqdm import tqdm

import cv2
import cmapy
import librosa
import torch
import torchaudio
from torchaudio import transforms as T
from scipy.signal import butter, lfilter

from .augmentation import augment_raw_audio

__all__ = ['get_annotations', 'save_image', 'get_mean_and_std', 'get_individual_cycles_librosa', 'get_individual_cycles_torchaudio', 'split_pad_sample', 'generate_mel_spectrogram', 'generate_fbank', 'concat_augmentation', 'get_score']


# ==========================================================================
""" ICBHI dataset information """
def _extract_lungsound_annotation(file_name, data_folder):
    tokens = file_name.strip().split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient Number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_folder, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')

    return recording_info, recording_annotations


def get_annotations(args, data_folder):
    if args.class_split == 'lungsound' or args.class_split in ['lungsound_meta', 'meta']:
        filenames = [f.strip().split('.')[0] for f in os.listdir(data_folder) if '.txt' in f]

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            annotation_dict[f] = ann

    elif args.class_split == 'diagnosis':
        filenames = [f.strip().split('.')[0] for f in os.listdir(data_folder) if '.txt' in f]
        tmp = pd.read_csv(os.path.join(args.data_folder, 'icbhi_dataset/patient_diagnosis.txt'), names=['Disease'], delimiter='\t')

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            ann.drop(['Crackles', 'Wheezes'], axis=1, inplace=True)

            disease = tmp.loc[int(f.strip().split('_')[0]), 'Disease']
            ann['Disease'] = disease

            annotation_dict[f] = ann
            
    return annotation_dict


def save_image(image, fpath):
    save_dir = os.path.join(fpath, 'image.jpg')
    cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    

def get_mean_and_std(dataset):
    """ Compute the mean and std value of mel-spectrogram """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    cnt = 0
    fst_moment = torch.zeros(1)
    snd_moment = torch.zeros(1)
    for inputs, _, _ in dataloader:
        b, c, h, w = inputs.shape
        nb_pixels = b * h * w

        fst_moment += torch.sum(inputs, dim=[0,2,3])
        snd_moment += torch.sum(inputs**2, dim=[0,2,3])
        cnt += nb_pixels

    mean = fst_moment / cnt
    std = torch.sqrt(snd_moment/cnt - mean**2)

    return mean, std
# ==========================================================================


# ==========================================================================
""" data preprocessing """
def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    return y


def _slice_data_librosa(start, end, data, sample_rate):
    """
    RespireNet paper..
    sample_rate denotes how many sample points for one second
    """
    max_ind = len(data) 
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[start_ind: end_ind]


def _get_lungsound_label(crackle, wheeze, n_cls):
    if n_cls == 4:
        if crackle == 0 and wheeze == 0:
            return 0
        elif crackle == 1 and wheeze == 0:
            return 1
        elif crackle == 0 and wheeze == 1:
            return 2
        elif crackle == 1 and wheeze == 1:
            return 3
    
    elif n_cls == 2:
        if crackle == 0 and wheeze == 0:
            return 0
        else:
            return 1


def _get_diagnosis_label(disease, n_cls):
    if n_cls == 3:
        if disease in ['COPD', 'Bronchiectasis', 'Asthma']:
            return 1
        elif disease in ['URTI', 'LRTI', 'Pneumonia', 'Bronchiolitis']:
            return 2
        else:
            return 0

    elif n_cls == 2:
        if disease == 'Healthy':
            return 0
        else:
            return 1


def get_individual_cycles_librosa(args, recording_annotations, data_folder, filename, sample_rate, n_cls, butterworth_filter=None):
    """
    RespireNet paper..
    Used to split each individual sound file into separate sound clips containing one respiratory cycle each
    output: [(audio_chunk:np.array, label:int), (...)]
    """
    sample_data = []

    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(data_folder, filename+'.wav'), sr=sample_rate)

    if butterworth_filter:
        # butter bandpass filter
        data = _butter_bandpass_filter(lowcut=200, highcut=1800, fs=sample_rate, order=butterworth_filter)
    
    for idx in recording_annotations.index:
        row = recording_annotations.loc[idx]

        start = row['Start'] # time (second)
        end = row['End'] # time (second)
        audio_chunk = _slice_data_librosa(start, end, data, rate)

        if args.class_split == 'lungsound':
            crackles = row['Crackles']
            wheezes = row['Wheezes']            
            sample_data.append((audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls)))
        elif args.class_split == 'diagnosis':
            disease = row['Disease']            
            sample_data.append((audio_chunk, _get_diagnosis_label(disease, n_cls)))

    return sample_data


def _slice_data_torchaudio(start, end, data, sample_rate):
    """
    SCL paper..
    sample_rate denotes how many sample points for one second
    """
    max_ind = data.shape[1]
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[:, start_ind: end_ind]


def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data

def get_individual_cycles_torchaudio(args, recording_annotations, metadata, data_folder, filename, sample_rate, n_cls):
    """
    SCL paper..
    used to split each individual sound file into separate sound clips containing one respiratory cycle each
    output: [(audio_chunk:np.array, label:int), (...)]
    """
    sample_data = []
    fpath = os.path.join(data_folder, filename+'.wav')
        
    sr = librosa.get_samplerate(fpath)
    data, _ = torchaudio.load(fpath)
    
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        data = resample(data)

    fade_samples_ratio = 16
    fade_samples = int(sample_rate / fade_samples_ratio)

    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')

    data = fade(data)
    for idx in recording_annotations.index:
        row = recording_annotations.loc[idx]

        start = row['Start'] # time (second)
        end = row['End'] # time (second)
        audio_chunk = _slice_data_torchaudio(start, end, data, sample_rate)

        if args.class_split == 'lungsound':
            crackles = row['Crackles']
            wheezes = row['Wheezes']            
            sample_data.append((audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls)))
        elif args.class_split == 'diagnosis':
            disease = row['Disease']            
            sample_data.append((audio_chunk, _get_diagnosis_label(disease, n_cls)))

    padded_sample_data = []
    for data, label in sample_data:
        data = cut_pad_sample_torchaudio(data, args)
        padded_sample_data.append((data, label))

    return padded_sample_data


def _zero_padding(source, output_length):
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)

    frac = src_length / output_length
    if frac < 0.5:
        # tile forward sounds to fill empty space
        cursor = 0
        while(cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        # [src_length:] part will be zeros
        copy[:src_length] = source[:]

    return copy


def _equally_slice_pad_sample(sample, desired_length, sample_rate):
    """
    pad_type == 0: zero-padding
    if sample length > desired_length, 
    all equally sliced samples with samples_per_slice number are zero-padded or recursively duplicated
    """
    output_length = int(desired_length * sample_rate) # desired_length is second
    soundclip = sample[0].copy()
    n_samples = len(soundclip)

    total_length = n_samples / sample_rate # length of cycle in seconds
    n_slices = int(math.ceil(total_length / desired_length)) # get the minimum number of slices needed
    samples_per_slice = n_samples // n_slices

    output = [] # holds the resultant slices
    src_start = 0 # staring index of the samples to copy from the sample buffer
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start

        copy = _zero_padding(soundclip[src_start:src_end], output_length)
        output.append((copy, sample[1], sample[2]))
        src_start += length

    return output


def _duplicate_padding(sample, source, output_length, sample_rate, types):
    # pad_type == 1 or 2
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    left = output_length - src_length # amount to be padded

    if types == 'repeat':
        aug = sample
    else:
        aug = augment_raw_audio(sample, sample_rate)

    while len(aug) < left:
        aug = np.concatenate([aug, aug])

    prob = random.random()
    if prob < 0.5:
        # pad the back part of original sample
        copy[left:] = source
        copy[:left] = aug[len(aug)-left:]
    else:
        # pad the front part of original sample
        copy[:src_length] = source[:]
        copy[src_length:] = aug[:left]

    return copy


def split_pad_sample(sample, desired_length, sample_rate, types='repeat'):
    """
    if the audio sample length > desired_length, then split and pad samples
    else simply pad samples according to pad_types
    * types 'zero'   : simply pad by zeros (zero-padding)
    * types 'repeat' : pad with duplicate on both sides (half-n-half)
    * types 'aug'    : pad with augmented sample on both sides (half-n-half)	
    """
    if types == 'zero':
        return _equally_slice_pad_sample(sample, desired_length, sample_rate)

    output_length = int(desired_length * sample_rate)
    soundclip = sample[0].copy()
    n_samples = len(soundclip)

    output = []
    if n_samples > output_length:
        """
        if sample length > desired_length, slice samples with desired_length then just use them,
        and the last sample is padded according to the padding types
        """
        # frames[j] = x[j * hop_length : j * hop_length + frame_length]
        frames = librosa.util.frame(soundclip, frame_length=output_length, hop_length=output_length//2, axis=0)
        for i in range(frames.shape[0]):
            output.append((frames[i], sample[1], sample[2]))

        # get the last sample
        last_id = frames.shape[0] * (output_length//2)
        last_sample = soundclip[last_id:]
        
        padded = _duplicate_padding(soundclip, last_sample, output_length, sample_rate, types)
        output.append((padded, sample[1], sample[2]))
    else: # only pad
        padded = _duplicate_padding(soundclip, soundclip, output_length, sample_rate, types)
        output.append((padded, sample[1], sample[2]))

    return output


def generate_mel_spectrogram(audio, sample_rate, n_mels=64, f_min=50, f_max=2000, nfft=1024, hop=512, args=None):
    """
    use librosa library and convert mel-spectrogram to have 3 channels
    """
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    # convert scale to dB from magnitude
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min())
    # S *= 255

    if args.blank_region_clip:
        S = cv2.flip(S, 0) # up-down flip
    img = S.reshape(S.shape[0], S.shape[1], 1)

    return img


def generate_fbank(audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    mean, std =  -4.2677393, 4.5689974
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank 


def concat_augmentation(classwise_cycle_list, cycle_list, scale=1.):
    """ From "RespireNet" paper..
    """

    def _get_random_cycles(classwise_cycle_list, idx1, idx2):
        i = random.randint(0, len(classwise_cycle_list[idx1])-1)
        j = random.randint(0, len(classwise_cycle_list[idx2])-1)

        sample_i = classwise_cycle_list[idx1][i]
        sample_j = classwise_cycle_list[idx2][j]

        return sample_i, sample_j

    print('*' * 20)
    # augment normal
    aug_nums = int(scale*len(classwise_cycle_list[0]) - len(classwise_cycle_list[0]))
    print('# of concatenation-based augmentation for normal  class is {}'.format(aug_nums))

    for _ in range(aug_nums):
        sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 0, 0)
        new_sample = np.concatenate([sample_i[0], sample_j[0]])
        # cycle_list: [(audio_chunk, label, filename, pad_times), (...)]
        cycle_list.append((new_sample, 0, sample_i[2]+'-'+sample_j[2])) # sample_i[2] denotes filename
    
    # augment crackle
    aug_nums = int(scale*len(classwise_cycle_list[0]) - len(classwise_cycle_list[1]))
    print('# of concatenation-based augmentation for crackle class is {}'.format(aug_nums))

    for _ in range(aug_nums):
        aug_prob = random.random()
        if aug_prob < 0.6:
            # crackle_i + crackle_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 1, 1)
        elif aug_prob >= 0.6 and aug_prob < 0.8:
            # crackle_i + normal_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 1, 0)
        else:
            # normal_i + crackle_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 0, 1)

        new_sample = np.concatenate([sample_i[0], sample_j[0]])
        cycle_list.append((new_sample, 1, sample_i[2]+'-'+sample_j[2]))
    
    # augment wheeze
    aug_nums = int(scale*len(classwise_cycle_list[0]) - len(classwise_cycle_list[2]))
    print('# of concatenation-based augmentation for wheeze  class is {}'.format(aug_nums))

    for _ in range(aug_nums):
        aug_prob = random.random()
        if aug_prob < 0.6:
            # wheeze_i + wheeze_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 2, 2)
        elif aug_prob >= 0.6 and aug_prob < 0.8:
            # wheeze_i + normal_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 2, 0)
        else:
            # normal_i + wheeze_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 0, 2)

        new_sample = np.concatenate([sample_i[0], sample_j[0]])
        cycle_list.append((new_sample, 2, sample_i[2]+'-'+sample_j[2]))

    # augment both
    aug_nums = int(scale*len(classwise_cycle_list[0]) - len(classwise_cycle_list[3]))
    print('# of concatenation-based augmentation for both   class is {}'.format(aug_nums))

    for _ in range(aug_nums):
        aug_prob = random.random()
        if aug_prob < 0.5:
            # both_i + both_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 3, 3)
        elif aug_prob >= 0.5 and aug_prob < 0.7:
            # crackle_i + wheeze_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 1, 2)
        elif aug_prob >=0.7 and aug_prob < 0.8:
            # wheeze_i + crackle_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 2, 1)
        elif aug_prob >=0.8 and aug_prob < 0.9:
            # both_i + normal_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 3, 0)
        else:
            # normal_i + both_j
            sample_i, sample_j = _get_random_cycles(classwise_cycle_list, 0, 3)

        new_sample = np.concatenate([sample_i[0], sample_j[0]])
        cycle_list.append((new_sample, 3, sample_i[2]+'-'+sample_j[2]))

    return classwise_cycle_list, cycle_list
# ==========================================================================


# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================
