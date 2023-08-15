import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def get_act_fn(name):
    if name == 'relu':
        fn = nn.ReLU
    elif name == 'gelu':
        fn = nn.GELU
    elif name == 'prelu':
        fn = nn.PReLU
    elif name == 'tanh':
        fn = nn.Tanh
    elif name == 'leakyrelu':
        fn = nn.LeakyReLU
    elif name =='sigmoid':
        fn = nn.Sigmoid
    else:
        raise ValueError("Unsupported activation function: {}".format(name))
    return fn


def get_loss_fn(name):
    if name == 'mse':
        fn = nn.MSELoss
    elif name == 'l1':
        fn = nn.L1Loss
    elif name == 'l1smooth':
        fn = nn.SmoothL1Loss
    else:
        raise ValueError("Unsupported loss function: {}".format(name))
    return fn

def get_reduce_fn(name):
    if name == 'mean':
        fn = torch.mean
    elif name == 'sum':
        fn = torch.sum
    else:
        raise
    return fn

def gather_on_indices(pidx, hypotheses):
    batch_list = torch.arange(hypotheses.size(0))
    selected_hypo = hypotheses[batch_list, pidx, :]
    return selected_hypo


def sample_predictor(coef_logits, mix_sampling_policy, temperature=0.3):
    '''
    Select predictor according to sampling policy.
    '''
    if mix_sampling_policy == 'greedy':
        # Predictor with highest coefficient is used for inference.
        predictor_idx = torch.argmax(coef_logits, dim=-1)

    elif mix_sampling_policy == 'multi':
        # According to paper
        predictor_idx = Categorical(logits=coef_logits / temperature).sample()

    else:
        raise ValueError("Not implemented f{mix_sampling_policy}")

    return predictor_idx