############ Adapted from  https://github.com/moskomule/dda #############


""" Operations

"""
from scipy.special import logit
import numpy as np
import torch
from torch import nn
import random
from torch.distributions import RelaxedBernoulli, Bernoulli

from functional import *

import warp_ops


class _Operation(nn.Module):
    """ Base class of operation

    :param operation:
    :param initial_magnitude:
    :param initial_probability:
    :param learn_magnitude:
    :param learn_probability:
    :param temperature: Temperature for RelaxedBernoulli distribution used during training
    """

    def __init__(self,
                 operation,
                 initial_magnitude,
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        super().__init__()
        self.operation = operation

        if initial_magnitude is not None and learn_magnitude:
            self.magnitude = nn.Parameter(torch.Tensor(initial_magnitude))
        else:
            self.magnitude = torch.Tensor(initial_magnitude)

        if learn_probability:
            self.probability = nn.Parameter(torch.Tensor(
                [float(logit(i)) for i in initial_probability]))
        else:
            self.probability = torch.Tensor(
                [float(logit(i)) for i in initial_probability])

        assert 0 < temperature
        self.temperature = temperature

    def forward(self, input, label):
        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor.
        # First repeat in BS dimension
        BS, C, L = input.shape
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries.
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel)
        mask = mask.view(BS, 1, 1)
        freqmasked = (mask * transformed + (1 - mask) * input)
        return freqmasked

    def get_mask(self, label,
                 batch_size=None):

        prob = torch.sigmoid(self.probability).unsqueeze(0)
        prob = prob.repeat(batch_size, 1)
        prob = 0*prob.sum() + prob[torch.arange(batch_size), label.long()]
        if self.training:
            return RelaxedBernoulli(self.temperature, prob).rsample()
        else:
            return Bernoulli(prob).sample()


class NoOp(_Operation):
    def __init__(self,
                 initial_magnitude=[0., 0.],
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        super().__init__(None, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

    def forward(self, input, label):
        print('no op')
        return input

 
# (1) RandTemporalWarp
class RandTemporalWarp(_Operation):
    def __init__(self,
                 initial_magnitude=[0.9, 0.9],
                 # MI   [0.6, 0.6 - 0.725]
                 # STTC [0.9, 0.9 - 0.759]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(rand_temporal_warp, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

        # create the warp obj here.
        self.warp_obj = warp_ops.RandWarpAug([2496])

    def forward(self, input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor.
        # First repeat in BS dimension
        BS, C, L = input.shape
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries.
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel, self.warp_obj)
        B, C, L = transformed.shape
        mask = mask.view(B, 1, 1)
        freqmasked = (mask * transformed + (1 - mask) * input)
        return freqmasked

 
# (2) BaselineWander
class BaselineWander(_Operation):
    def __init__(self,
                 initial_magnitude=[1, 1],
                 # MI   [1., 1.] *0.8 - 0.717
                 # STTC [1., 1.] * 2  - 0.766
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(baseline_wander, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

 
# (3) GaussianNoise
class GaussianNoise(_Operation):
    def __init__(self,
                 initial_magnitude=[3., 3.],
                 # MI   [0.7 * 0.25 * [3., 3.] -- 0.717]
                 # STTC [0.5 * [3., 3.]        -- 0.758]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(gaussian_noise, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

 
# (4) RandCrop
class RandCrop(_Operation):
    def __init__(self,
                 # mag represents the cut percentage
                 initial_magnitude=[0.3],
                 # MI   [0.13 -- 0.706]
                 # STTC [0.1  -- 0.776]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(rand_crop, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

    def forward(self, input, label):
        mag = self.magnitude.to(input.device)
        transformed = self.operation(input, mag)
        return transformed

 
# (5) RandDisplacement
class RandDisplacement(_Operation):
    def __init__(self,
                 initial_magnitude=[0.18, 0.18],
                 # MI   [0.01, 0.01 -- 0.691]
                 # STTC [0.1, 0.1   -- 0.769]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(rand_displacement, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

    # create the warp obj here.
        self.warp_obj = warp_ops.DispAug([2496])

    def forward(self, input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor.
        # First repeat in BS dimension
        BS, C, L = input.shape
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries.
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel, self.warp_obj)
        B, C, L = transformed.shape
        mask = mask.view(B, 1, 1)
        freqmasked = (mask * transformed + (1 - mask) * input)
        return freqmasked

  
# (6) MagnitudeScale
class MagnitudeScale(_Operation):
    def __init__(self,
                 initial_magnitude=[0.6, 0.6],
                 # MI   [0.5, 0.5  -- 0.713]
                 # STTC [0.6, 0.6  -- 0.773]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(magnitude_scale, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

  
# (7) TimeMask
class TimeMask(_Operation):  # Time mask
    def __init__(self,
                 initial_magnitude=[0.01],  # mask percentage
                 # MI   [0.5   -- 0.717]
                 # STTC [0.01  -- 0.770]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(time_mask, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

    def forward(self, input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device)
        transformed = self.operation(input, mag)
        B, C, L = transformed.shape
        mask = mask.view(B, 1, 1)
        freqmasked = (mask * transformed + (1 - mask) * input)
        return freqmasked

  
# (8) Channel Mask
class ChannelMask(_Operation):
    def __init__(self,
                 initial_magnitude=[0.0, 0.0],
                 # MI   [num_mask = 7  --0.729]
                 # STTC [num_mask = 10 --0.805]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(channel_mask, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)

  
# (9) Permute Wave Segment
class PermuteWaveSegment(_Operation):
    def __init__(self,
                 initial_magnitude=[0.0, 0.0],
                 # MI   [8/0.3 --0.727]
                 # STTC [6/0.5 --0.778]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(permute_wave_segment, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)


 
# (10) Spectrum Mask
class Specmask(_Operation):
    def __init__(self,
                 initial_magnitude=[0.6, 0.6],
                 # MI   [0.26, 0.26 -- 0.710]
                 # STTC [0.5 , 0.5  -- 0.784]
                 # CD
                 # HYP
                 initial_probability=[0.9999999, 0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature=0.1,
                 ):
        print(f"initial magnitude: {initial_magnitude}")
        super().__init__(spec_mask, initial_magnitude, initial_probability, learn_magnitude,
                         learn_probability, temperature)
