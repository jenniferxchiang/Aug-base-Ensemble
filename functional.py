import torch.fft
import numpy as np
from torch.nn import functional as F
import librosa
import torch
import random
from torch.autograd import Function
import math
import random

 
# (1) RandTemporalWarp

def rand_temporal_warp(x, mag, warp_obj):
    mag = 100*(mag**2)
    print('in (1) rand temporal warp')  # initial_magnitude=[2., 2.]
    return warp_obj(x, mag)

 
# (2) BaselineWander
def baseline_wander(x, mag):
    BS, C, L = x.shape

    # form baseline drift
    strength = 0.3 * torch.sigmoid(mag) * \
        (torch.rand(BS).to(x.device).view(BS, 1, 1))
    # strength = 0.25 *torch.sigmoid(mag) * \
    #     (torch.rand(BS).to(x.device).view(BS, 1, 1))
    strength = strength.view(BS, 1, 1)
    print('in (2) baseline wander')  # initial_magnitude=[0.0, 0.0]
    frequency = ((torch.rand(BS) * 20 + 10) * 10 / 60).view(BS,
                                                            1, 1)  # typical breaths per second for an adult
    phase = (torch.rand(BS) * 2 * np.pi).view(BS, 1, 1)
    drift = strength*torch.sin(torch.linspace(0, 1, L).view(1, 1, -1)
                               * frequency.float() + phase.float()).to(x.device)
    return x + drift

 
# (3) GaussianNoise
def gaussian_noise(x, mag):
    BS, C, L = x.shape
    print('in (3) gaussian noise')  # initial_magnitude=[0.0, 0.0]
    stdval = torch.std(x, dim=2).view(BS, C, 1).detach()
    noise = 0.15 * stdval * \
        torch.sigmoid(mag)*torch.randn(BS, C, L).to(x.device)
    # noise = 0.25*stdval*torch.sigmoid(mag)*torch.randn(BS, C, L).to(x.device)
    return x + noise

 
# (4) RandCrop
def rand_crop(x, mag):
    x_aug = x.clone()
    # get shapes
    BS, C, L = x.shape
    mag = mag.item()

    crop_ratio = 1 - mag
    crop_length = int(L * crop_ratio)
    start = 10  # random.randint(0, L - crop_length)
    end = start + crop_length
    print(start)
    print(crop_length)
    print(end)
    print(L)
    # Concatenate the cropped segments
    x_aug = torch.cat((x_aug[:, :, end:], x_aug[:, :, :end]), dim=2)
    print('in (4) random crop')
    # Fill the cropped region with the circular extension of itself
    for i in range(BS):
        for j in range(C):
            left = start - L
            right = end - L
            padded_left = x_aug[i, j, L+left:] if left < 0 else torch.Tensor()
            padded_right = x_aug[i, j, :right] if right > 0 else torch.Tensor()
            if len(padded_left) > 0 and len(padded_right) > 0:
                x_aug[i, j, start:end] = torch.cat(
                    (padded_left, x_aug[i, j, start:end], padded_right), dim=0)
            elif len(padded_left) > 0:
                x_aug[i, j, start:end] = padded_left[-len(
                    x_aug[i, j, start:end]):] + x_aug[i, j, start:end]
            elif len(padded_right) > 0:
                x_aug[i, j, start:end] = x_aug[i, j, start:end] + \
                    padded_right[:len(x_aug[i, j, start:end])]
    return x_aug

 
# (5) RandDisplacement
def rand_displacement(x, mag, warp_obj):
    disp_mag = 100*(mag**2)  # initial_magnitude=[0.5, 0.5]
    print('in (5) random displacement')
    return warp_obj(x, disp_mag)

 
# (6) MagnitudeScale
def magnitude_scale(x, mag):
    BS, C, L = x.shape
    print('in (6) magnitude scale')  # initial_magnitude=[0.0, 0.0]
    strength = mag*(-0.5 * (torch.rand(BS).to(x.device)).view(BS, 1, 1) + 1.25)
    # strength = torch.sigmoid(
    #     mag)*(-0.5 * (torch.rand(BS).to(x.device)).view(BS, 1, 1) + 1.25)
    strength = strength.view(BS, 1, 1)
    return x*strength

 
# (7) TimeMask
def time_mask(x, mag):
    x_aug = x.clone()
    # get shapes
    BS, C, L = x.shape
    mag = mag.item()
    print('in (7) time mask')
    nmf = int(mag*L)  # initial_magnitude=[0.05]
    start = torch.randint(0, L-nmf, [1]).long()
    end = (start + nmf).long()
    x_aug[:, :, start:end] = 0.
    return x_aug

 
# (8) Channel Mask
def channel_mask(x, mag):
    num_masks = 1
    replace_with_zero = True
    cloned = x.clone()
    batch_size, num_leads, len_spectro = cloned.shape

    # Generate random indices for the masked leads
    mask_indices = torch.randperm(num_leads)[:num_masks]
    print("the masked leads:", mask_indices.tolist())

    for i in range(num_masks):
        if replace_with_zero:
            print('in (8) channel mask ')
            # Mask the selected lead only in the time range defined by f_zero and mask_end
            cloned[:, mask_indices[i], :] = 0
        else:
            cloned[:, mask_indices[i], :] = cloned.mean()
    return cloned

 
# (9) Permute Wave Segment
def permute_wave_segment(x, mag):
    num_segments = 12
    T = math.floor(2496 / num_segments)
    replace_percentage = 0.9
    cloned = x.clone()
    batch_size, num_leads, len_spectro = cloned.shape
    print(replace_percentage)
    print('in (9) permute wave segment ')

    for i in range(num_segments):
        start_idx = i * T
        end_idx = start_idx + T
        segment = cloned[:, :, start_idx:end_idx]

        # Randomly select the segment to replace
        replace_size = int(T * replace_percentage)
        replace_start = torch.randint(0, T - replace_size + 1, (batch_size,))
        replace_end = replace_start + replace_size

        # Generate a replacement mask
        replace_mask = torch.zeros((batch_size, T))
        for b in range(batch_size):
            replace_mask[b, replace_start[b]:replace_end[b]] = 1
        permutation = torch.randperm(batch_size)

        # Replace the selected segments
        for t in range(T):
            if replace_mask[permutation, t].sum() > 0:
                permuted_segment = segment[permutation, :, :]
                segment[:, :, t] = permuted_segment[:, :, t]

        # Merge the replaced segment back to the original waveform
        cloned[:, :, start_idx:end_idx] = segment

    return cloned

 
# Spec mask
def spec_mask(x, mag):
    num_ch = 12
    x_aug = x.clone()
    BS, C, L = x.shape
    mag = mag.view(-1)[0]
    print('in spec aug')
    # get shapes
    BS, NF, NT, _ = torch.stft(x[:, 0, ], n_fft=512, hop_length=4).shape
    nmf = int(mag*NF)
    start = torch.randint(0, NF-nmf, [1]).long()
    print('start spec mask:', start)
    end = (start + nmf).long()
    print('end spec mask:', end)
    for i in range(num_ch):
        stft_inp = torch.stft(x[:, i, ], n_fft=512, hop_length=4)
        stft_inp[:, start:end, :] = 0
        x_aug[:, i] = torch.istft(stft_inp, n_fft=512, hop_length=4)

    # nmf = int(mag*L)
    # start = torch.randint(0, L-nmf, [1]).long()
    # end = (start + nmf).long()
    # noise = torch.zeros_like(x)
    # noise[:, :, start:end] = 1
    # x_aug *= noise
    return x_aug
