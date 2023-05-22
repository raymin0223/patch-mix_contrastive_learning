import random
import numpy as np
import nlpaug.augmenter.audio as naa

import torch
from torchvision.utils import _log_api_usage_once
from torchvision.transforms import transforms
from torchaudio import transforms as T
from .time_warping import sparse_image_warp

__all__ = ['augment_raw_audio', 'SpecAugment']


def augment_raw_audio(sample, sample_rate, args):
    """
    Raw audio data augmentation technique
    you can utilize any library code
    1) nlpaug
    2) audiomentations
    3) librosa
    """

    """ 1) nlpaug """
    augment_list = [
        # naa.CropAug(sampling_rate=sample_rate)
        naa.NoiseAug(), # apply noise injection operation
        naa.SpeedAug(), # apply speed adjustment operation
        naa.LoudnessAug(factor=(0.5, 2)), # apply adjusting loudness operation
        naa.VtlpAug(sampling_rate=sample_rate, zone=(0.0, 1.0)), # apply vocal tract length perturbation (VTLP) operation
        naa.PitchAug(sampling_rate=sample_rate, factor=(-1,3)) # apply pitch adjustment operation
    ]

    # randomly sample augmentation
    aug_idx = random.randint(0, len(augment_list)-1)
    sample = augment_list[aug_idx].augment(sample)

    # apply all augmentations
    # for aug_idx in range(len(augment_list)):
    #     sample = augment_list[aug_idx].augment(sample)
    
    """ 2) audiomentations """
    # import audiomentations
    # from audiomentations import AddGaussianSNR, TimeStretch, PitchShift, Shift

    # # when using audiomentations library (not DEBUG yet)
    # audio_transforms = audiomentations.Compose([
    #     AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=40.0, p=0.5),
    #     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    #     PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    #     Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    # ])

    # sample = audio_transforms(samples=sample, sample_rate=sample_rate)

    """ 3) librosa """
    # import librosa

    # def _noise(data):
    #     noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    #     data = data + noise_amp * np.random.normal(size=data.shape[0])
    #     return data

    # def _stretch(data, rate=0.8):
    #     return librosa.effects.time_stretch(data, rate)

    # def _shift(data):
    #     shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    #     return np.roll(data, shift_range)
        
    # def _pitch(data, sampling_rate, pitch_factor=0.7):
    #     return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    # sample = _noise(sample)
    # sample = _stretch(sample)
    # sample = _shift(sample)
    # sample = _pitch(sample, sample_rate)

    if type(sample) == list:
        return sample[0]
    else:
        return sample


# Use this Class when you load dataset with librosa
class SpecAugment(torch.nn.Module):
    '''
    Unofficial Implementation of SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    Paper: https://arxiv.org/pdf/1904.08779.pdf
    Ref. github: https://github.com/pyyush/SpecAugment/blob/219fc6e9ed4838fe9700295700040b1da283c536/augment.py#L10

    Augmentation Parameters for policies
    -----------------------------------------
    Policy | W  | F  | m_F |  T  |  p  | m_T
    -----------------------------------------
    None   |  0 |  0 |  -  |  0  |  -  |  -
    -----------------------------------------
    LB     | 80 | 27 |  1  | 100 | 1.0 | 1
    -----------------------------------------
    LD     | 80 | 27 |  2  | 100 | 1.0 | 2
    -----------------------------------------
    SM     | 40 | 15 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    SS     | 40 | 27 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    
    LB  : LibriSpeech basic
    LD  : LibriSpeech double
    SM  : Switchboard mild
    SS  : Switchboard strong
    W   : Time Warp parameter
    F   : Frequency Mask parameter
    m_F : Number of Frequency masks
    T   : Time Mask parameter
    p   : Parameter for calculating upper bound for time mask
    m_T : Number of time masks
    '''
    #def __init__(self, policy, zero_mean_normalized=False):
    def __init__(self, args):
        super().__init__()
        _log_api_usage_once(self)

        self.policy = args.specaug_policy
        self.mask = args.specaug_mask
        
        # Policy Specific Parameters
        if self.policy == 'LB':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif self.policy == 'LD':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif self.policy == 'SM':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif self.policy == 'SS':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2
        elif self.policy == 'icbhi_sup':
            # following https://github.com/ilyassmoummad/scl_icbhi2017
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 0, 20, 2, 50, 1.0, 2
        elif self.policy == 'icbhi_ast_sup':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 0, 48, 2, 160, 1.0, 2

        # mimic SpecAugment by using torchaudio.transforms
        # self.spec_aug = torch.nn.Sequential(
        #     T.TimeStretch(0.8, fixed_rate=True),
        #     T.FrequencyMasking(freq_mask_param=self.F),
        #     T.TimeMasking(time_mask_param=self.T)
        # )
                
    def time_warp(self):   
        """ Tensorflow version """ 
        # v, tau = self.mel_spectrogram.shape[1], self.mel_spectrogram.shape[2]
        
        # horiz_line_thru_ctr = self.mel_spectrogram[0][v//2]
    
        # random_pt = horiz_line_thru_ctr[random.randrange(self.W, tau - self.W)] # random point along the horizontal/time axis
        # w = np.random.uniform((-self.W), self.W) # distance
        
        # src_points = [[[v//2, random_pt[0]]]] # Source Points
        # dest_points = [[[v//2, random_pt[0] + w]]] # Destination Points
        # self.mel_spectrogram, _ = sparse_image_warp(self.mel_spectrogram, src_points, dest_points, num_boundary_points=2)
        # self.mel_spectrogram = self.mel_spectrogram.numpy()

        """ Pytorch version """
        # refer to https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
        num_rows = self.mel_spectrogram.shape[2]
        spec_len = self.mel_spectrogram.shape[1]
        device = self.mel_spectrogram.device

        # adapted from https://github.com/DemisEom/SpecAugment/
        pt = (num_rows - 2 * self.W) * torch.rand([1], dtype=torch.float) + self.W # random point along the time axis
        src_ctr_pt_freq = torch.arange(0, spec_len // 2)  # control points on freq-axis
        src_ctr_pt_time = torch.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
        src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1)
        src_ctr_pts = src_ctr_pts.float().to(device)

        # Destination
        w = 2 * self.W * torch.rand([1], dtype=torch.float) - self.W # distance
        dest_ctr_pt_freq = src_ctr_pt_freq
        dest_ctr_pt_time = src_ctr_pt_time + w
        dest_ctr_pts = torch.stack((dest_ctr_pt_freq, dest_ctr_pt_time), dim=-1)
        dest_ctr_pts = dest_ctr_pts.float().to(device)

        # warp
        source_control_point_locations = torch.unsqueeze(src_ctr_pts, 0)  # (1, v//2, 2)
        dest_control_point_locations = torch.unsqueeze(dest_ctr_pts, 0)  # (1, v//2, 2)
        warped_spectro, dense_flows = sparse_image_warp(self.mel_spectrogram, source_control_point_locations, dest_control_point_locations)

        return warped_spectro.squeeze(3)

    def freq_mask(self):
        if self.mask == 'mean':
            # maksing to mean value
            mask_value = self.mel_spectrogram.mean()
        elif self.mask == 'zero':
            # maksing to zero value
            mask_value = 0.

        v = self.mel_spectrogram.shape[1] # no. of mel bins
        
        # apply m_F frequency masks to the mel spectrogram
        for i in range(self.m_F):
            f = int(np.random.uniform(0, self.F)) # [0, F)
            f0 = random.randint(0, v - f) # [0, v - f)
            self.mel_spectrogram[:, f0:f0 + f, :] = mask_value
            
        return self.mel_spectrogram
        
    def time_mask(self):
        if self.mask == 'mean':
            # maksing to mean value
            mask_value = self.mel_spectrogram.mean()
        elif self.mask == 'zero':
            # maksing to zero value
            mask_value = 0.

        tau = self.mel_spectrogram.shape[2] # time frames
        
        # apply m_T time masks to the mel spectrogram
        for i in range(self.m_T):
            t = int(np.random.uniform(0, self.T)) # [0, T)
            t0 = random.randint(0, tau - t) # [0, tau - t)
            self.mel_spectrogram[:, :, t0:t0 + t] = mask_value
            
        return self.mel_spectrogram

    def forward(self, img):
        """
        Args:
            img (Tensor): Mel-spectrogram to be specaugmented.
        Returns:
            Tensor: Time-warped, time masked and freq masked image.
        """
        # self.mel_spectrogram = img # np.array [time, freq, channel]
        self.mel_spectrogram = img # torch.tensor [channel, time, freq]
        self.mel_spectrogram = self.mel_spectrogram.transpose(2, 1) # torch.tensor [channel, freq, time]

        if self.p >= torch.randn(1):
            if self.W:
                try:
                    # input shape of time_warp should be [sample_size, time, freq]
                    # assume that channel == 1 and augment each "one" sample
                    self.mel_spectrogram= self.time_warp()
                except Exception as e:
                    # torch.linalg.solve: (Batch element 0): The solver failed because the input matrix is singular.
                    # print(e)
                    pass

            self.mel_spectrogram = self.freq_mask()
            self.mel_spectrogram = self.time_mask()
        
        return self.mel_spectrogram.transpose(2, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"