from itertools import product

import torch
import torch.nn as nn


class MhdLayer(nn.Module):
    '''
    Multiple Hypothesis Dropout Layer
    '''
    def __init__(self, 
        inp_dim,
        subset_ratio=0.75,
        hypo_dim=1,
        wta_loss='vanilla',
        gate_sampling='exact'):
        super().__init__()
        assert gate_sampling in ['exact', 'approximate' ,'test'], f'{gate_sampling} not implemented. select from (exact, approximate)'
        assert wta_loss in ['vanilla','mixture'], f"{wta_loss} not implemented. select from (vanilla, mixture)"

        self.hypo_dim = hypo_dim
        self.wta_loss = wta_loss
        self.gate_sampling = gate_sampling

        self.all_gates = list(p for p in product(range(2), repeat=inp_dim) if sum(p) > 0) #list(product(range(2), repeat=inp_dim))
        self.gate_len = len(self.all_gates)

        self.subset_ratio = subset_ratio
        self.subset_size = max(1, int(round(self.subset_ratio * len(self.all_gates))))

    def forward(self, *args, **kwargs):

        if self.wta_loss == 'vanilla':
            return self._forward_vanilla( *args,**kwargs)
        elif self.wta_loss == 'mixture':
            return self._forward_mix( *args,**kwargs)

    def _forward_vanilla(self, x, hypo_count=None):

        x_shape = list(x.shape)
        bsz, inp_dim = x_shape

        if hypo_count is None or \
            hypo_count > 2 ** inp_dim:
            hypo_count = self.subset_size

        repeat_frame = [1] * (len(x_shape) + 1)
        repeat_frame[self.hypo_dim] = hypo_count

        # Repeat values across hypothesis dimension
        x = x.unsqueeze(self.hypo_dim).repeat(repeat_frame)

        gate_idx = self.sample_gate_idx(bsz, hypo_count=hypo_count, device=x.device)

        output = self.apply_gate(x, gate_idx)

        return output, gate_idx
    
    def _forward_mix(self, x, hypo_count=None):

        x_shape = list(x.shape)
        bsz, _ = x_shape

        hypo_count = self.subset_size

        repeat_frame = [1] * (len(x_shape) + 1)
        repeat_frame[self.hypo_dim] = hypo_count

        # Repeat values across hypothesis dimension
        x = x.unsqueeze(self.hypo_dim).repeat(repeat_frame)

        gate_idx = self.sample_gate_idx(bsz, hypo_count=hypo_count, device=x.device)

        # Apply Bernoulli variable
        output = self.apply_gate(x, gate_idx)
        
        return output, gate_idx
        
    
    def sample_gate_idx(self, bsz, hypo_count=None, device='cpu'):

        if hypo_count is None:
            hypo_count = self.subset_size

        if self.gate_sampling == 'exact':
            return torch.stack([torch.randperm(self.gate_len)[:hypo_count] for _ in range(bsz)], dim=0).to(device)
        elif self.gate_sampling == 'approximate':
            return torch.randint(0, self.gate_len, size=(bsz, hypo_count), device=device)
        elif self.gate_sampling == 'test':
            return torch.stack([torch.arange(self.gate_len)[:hypo_count] for _ in range(bsz)], dim=0).to(device)
        else:
            raise ValueError("Invalid gate sampling type given f{gate_sampling}")
    
    def apply_gate(self, x, gate_idx):
        return x * torch.tensor(self.all_gates, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)[gate_idx]

        