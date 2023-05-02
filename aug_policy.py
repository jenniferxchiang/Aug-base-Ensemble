############ Adapted from  https://github.com/moskomule/dda #############


import random
from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, RelaxedOneHotCategorical

from operations import *


class SubPolicyStage(nn.Module):
    def __init__(self,
                 operations,
                 temperature=0.05,
                 ):
        super(SubPolicyStage, self).__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature

    def forward(self,
                input, y):
        if self.training:
            relaxcat = RelaxedOneHotCategorical(torch.Tensor(
                [0.1]).to(input.device), logits=self._weights)
            wt = relaxcat.rsample()
            op_idx = wt.argmax().detach()
            op_mag = wt[op_idx] / wt[op_idx].detach()
            op_weights = torch.zeros(len(self.operations)).to(input.device)
            op_weights[op_idx] = op_mag
            return torch.stack([op_weights[i]*op(input, y) for i, op in enumerate(self.operations)]).sum(0)
        else:
            return self.operations[Categorical(logits=self._weights).sample()](input, y)

    @property
    def weights(self
                ):
        return self._weights.div(self.temperature).softmax(0)


class SubPolicy(nn.Module):
    def __init__(self, sub_policy_stage, operation_count=1):
        super().__init__()
        self.stages = nn.ModuleList(
            [deepcopy(sub_policy_stage) for _ in range(operation_count)])

    def forward(self, input, y):
        for stage in self.stages:
            input = stage(input, y)
        return input


class Policy(nn.Module):
    def __init__(self, operations, num_sub_policies=1, operation_count=2, num_chunks=1):
        super().__init__()
        self.sub_policies = nn.ModuleList([SubPolicy(SubPolicyStage(operations), operation_count)
                                           for _ in range(num_sub_policies)])
        # print(self.sub_policies)
        self.num_sub_policies = num_sub_policies
        self.operation_count = operation_count
        self.num_chunks = num_chunks

    def forward(self, x, y):
        x = self._forward(x, y)
        return x

    def _forward(self, input, y):
        index = 0  # random.randrange(self.num_sub_policies)
        # print("index:", index, "\n")

        # print(self.sub_policies[index], 'self.sub_policies[index]')
        return self.sub_policies[index](input, y)

# ==========================================================================================


def first_op(a, learn_mag=False, learn_prob=False):
    if a == "0":
        return [NoOp(), ]
    elif a == "1":
        return [
            RandTemporalWarp(learn_magnitude=learn_mag,
                             learn_probability=learn_prob),
        ]
    elif a == "2":
        return [
            BaselineWander(learn_magnitude=learn_mag,
                           learn_probability=learn_prob),
        ]
    elif a == "3":
        return [
            GaussianNoise(learn_magnitude=learn_mag,
                          learn_probability=learn_prob),
        ]
    elif a == "4":
        return [
            RandCrop(learn_probability=learn_prob),
        ]
    elif a == "5":
        return [
            RandDisplacement(learn_magnitude=learn_mag,
                             learn_probability=learn_prob),
        ]
    elif a == "6":
        return [
            MagnitudeScale(learn_magnitude=learn_mag,
                           learn_probability=learn_prob),
        ]
    elif a == "7":
        return [
            TimeMask(learn_magnitude=learn_mag,
                     learn_probability=learn_prob),
        ]
    elif a == "8":
        return [
            ChannelMask(learn_magnitude=learn_mag,
                        learn_probability=learn_prob),
        ]
    elif a == "9":
        return [
            PermuteWaveSegment(learn_magnitude=learn_mag,
                               learn_probability=learn_prob),
        ]
    elif a == "10":
        return [
            Specmask(learn_magnitude=learn_mag,
                     learn_probability=learn_prob),
        ]


def second_op(b, learn_mag=False, learn_prob=False):
    if b == "0":
        return [NoOp(), ]
    elif b == "1":
        return [
            RandTemporalWarp(learn_magnitude=learn_mag,
                             learn_probability=learn_prob),
        ]
    elif b == "2":
        return [
            BaselineWander(learn_magnitude=learn_mag,
                           learn_probability=learn_prob),
        ]
    elif b == "3":
        return [
            GaussianNoise(learn_magnitude=learn_mag,
                          learn_probability=learn_prob),
        ]
    elif b == "4":
        return [
            RandCrop(learn_probability=learn_prob),
        ]
    elif b == "5":
        return [
            RandDisplacement(learn_magnitude=learn_mag,
                             learn_probability=learn_prob),
        ]
    elif b == "6":
        return [
            MagnitudeScale(learn_magnitude=learn_mag,
                           learn_probability=learn_prob),
        ]
    elif b == "7":
        return [
            TimeMask(learn_magnitude=learn_mag,
                     learn_probability=learn_prob),
        ]
    elif b == "8":
        return [
            ChannelMask(learn_magnitude=learn_mag,
                        learn_probability=learn_prob),
        ]
    elif b == "9":
        return [
            PermuteWaveSegment(learn_magnitude=learn_mag,
                               learn_probability=learn_prob),
        ]
    elif b == "10":
        return [
            Specmask(learn_magnitude=learn_mag,
                     learn_probability=learn_prob),
        ]


def third_op(learn_mag=False, learn_prob=False):

    return [
        NoOp(),
    ]


# ==========================================================================================


def first_policy(a, num_sub_policies=1,  # stage ; big rectangle
                 operation_count=1,  # circle
                 num_chunks=1,
                 learn_mag=False, learn_prob=False):
    print('in first policy')
    return Policy(nn.ModuleList(first_op(a, learn_mag, learn_prob)), num_sub_policies,  operation_count,
                  num_chunks)


def second_policy(b, num_sub_policies=1,  # stage  rectangle
                  operation_count=1,  # circle(?)
                  num_chunks=1,
                  learn_mag=False, learn_prob=False):

    return Policy(nn.ModuleList(second_op(b, learn_mag, learn_prob)), num_sub_policies,  operation_count,
                  num_chunks)


def third_policy(num_sub_policies=1,  # stage  rectangle
                 operation_count=1,  # circle(?)
                 num_chunks=1,
                 learn_mag=False, learn_prob=False):

    return Policy(nn.ModuleList(third_op(learn_mag, learn_prob)), num_sub_policies,  operation_count,
                  num_chunks)
