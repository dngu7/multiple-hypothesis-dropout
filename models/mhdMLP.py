import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_loss_fn, gather_on_indices
from torch.distributions.categorical import Categorical

from models.mlp import MLP
from models.mhdLayer import MhdLayer

class MHDropoutMLP(nn.Module):
    '''
    Implements MH Dropout Network for MoM model (mixtureOM.py)
    '''
    def __init__(self, 
        inp_dim, 
        hid_dim, 
        out_dim,
        mhd_hid_dim=None, 
        wta_loss='vanilla',
        num_layers=1, 
        subset_ratio=0.5,
        distance_loss='mse', 
        act_fn='relu',
        out_act_fn=None,
        temperature=0.3,
        coef_weight=0.25,
        hypo_dim=1,
        mix_sampling_policy='multi',
        mix_loss_agg='sum',
        use_pre=True,
        gate_sampling='exact'
        ):
        super().__init__()

        self.hypo_dim = hypo_dim
        self.temperature = temperature
        self.coef_weight = coef_weight
        self.use_pre = use_pre
        self.out_dim = out_dim
        self.mhd_hid_dim = mhd_hid_dim

        if mhd_hid_dim is None:
            mhd_hid_dim = hid_dim


        num_layers = num_layers // 2

        self.pre_layer = MLP(inp_dim, hid_dim, mhd_hid_dim * 2, num_layers=num_layers, act_fn=act_fn)
        
        self.mhd_dropout = MhdLayer(mhd_hid_dim,  
                                          subset_ratio=subset_ratio, 
                                          hypo_dim=hypo_dim, 
                                          wta_loss=wta_loss,
                                          gate_sampling=gate_sampling)
        
        self.var_layer = MLP(mhd_hid_dim, mhd_hid_dim, mhd_hid_dim, num_layers=2, act_fn=act_fn)
        
        self.final_layer = MLP(mhd_hid_dim, hid_dim, out_dim, num_layers=num_layers, out_act_fn=out_act_fn, act_fn=act_fn)

        # Setup wta loss function
        assert wta_loss in ['vanilla','mixture'], f"{wta_loss} not implemented"
        self.wta_loss = wta_loss
        self.dist_loss_fn = get_loss_fn(distance_loss)(reduction='none')
        self.mix_loss_agg = mix_loss_agg
        self.mix_sampling_policy = mix_sampling_policy

        if self.wta_loss == 'mixture':
            self.mix_components = self.mhd_dropout.gate_len
            self.mix_layer = MLP(inp_dim, hid_dim, self.mix_components, num_layers=3)

    def loss(self, x, y, hypo_count=None):

        hypotheses, gate_idx = self.get_hypotheses(x, hypo_count=hypo_count)

        coef_logits = None
        if self.wta_loss == 'mixture':
            coef_logits = self.mix_layer(x) 
            coef_logits = torch.gather(coef_logits, 1, gate_idx)  

        loss_dist, loss_coeff, pidx_best = self.compute_loss(y, hypotheses, coef_logits)
        loss = loss_dist if loss_coeff is None else (loss_dist + loss_coeff * self.coef_weight)
        pred_best = gather_on_indices(pidx_best, hypotheses=hypotheses)
        
        log_loss = {'loss_dist': loss_dist.mean().detach().cpu().item()}
        if loss_coeff is not None:
            log_loss['loss_coeff'] = loss_coeff.mean().detach().cpu().item()
            
        return {'loss': loss, 'pred_best': pred_best, 'hypotheses': hypotheses}, log_loss
        
    def sample(self, x, hypo_count=None):
        output_dict = {}

        hypotheses, gate_idx = self.get_hypotheses(x, hypo_count=hypo_count)
        
        if self.wta_loss == 'mixture':
            coef_logits = self.mix_layer(x)
            coef_logits = torch.gather(coef_logits, 1, gate_idx)  

            pidx_sample = self.sample_predictor(coef_logits)
            pred_sample = gather_on_indices(pidx_sample, hypotheses=hypotheses)

            output_dict['coef_logits'] = coef_logits
            output_dict['pred_sample'] = pred_sample

        output_dict['hypotheses'] = hypotheses
        output_dict['sigma'] = torch.std(hypotheses, dim=1) 

        return output_dict
    
    def forward(self, x, y=None, hypo_count=None):

        output_dict = {}
        
        hypotheses, _ = self.get_hypotheses(x, hypo_count=hypo_count)
        coef_logits = None
        if self.wta_loss == 'mixture':
            coef_logits = self.mix_layer(x)

            if self.training:
                bsz = x.shape[0]
                gate_idx = self.mhd_dropout.sample_gate_idx(bsz)
                gate_idx = gate_idx.to(hypotheses.device)
                
                hypotheses = torch.gather(hypotheses, 1, gate_idx.unsqueeze(-1).repeat(1,1,self.out_dim))
                coef_logits = torch.gather(coef_logits, 1, gate_idx)
                
            pidx_sample = self.sample_predictor(coef_logits)
            pred_sample = gather_on_indices(pidx_sample, hypotheses=hypotheses)

            output_dict['coef_logits'] = coef_logits
            output_dict['pred_sample'] = pred_sample
            output_dict['pidx_sample'] = pidx_sample

        output_dict['hypotheses'] = hypotheses
        output_dict['sigma'] = torch.std(hypotheses, dim=1)

        if y is not None:

            loss, pidx_best = self.compute_loss(y, hypotheses, coef_logits)
            pred_best = gather_on_indices(pidx_best, hypotheses=hypotheses)

            output_dict['loss'] = loss
            output_dict['pidx_best'] = pidx_best
            output_dict['pred_best'] = pred_best
        
        return output_dict

    def get_hypotheses(self, x, hypo_count=None, dropout=True):
        encoded = self.pre_layer(x)

        mu_x = encoded[:, :self.mhd_hid_dim]
        sigma_x = encoded[:, self.mhd_hid_dim:]
        #print("mu_x", mu_x.shape)
        
        gate_idx = None
        if dropout:
            #print("sigma_x", sigma_x.shape)
            mhd_out, gate_idx = self.mhd_dropout(
                x=sigma_x, hypo_count=hypo_count)
            
            
            mhd_out = self.var_layer(mhd_out)

            mu_x = mu_x.unsqueeze(1).repeat(1,mhd_out.size(1),1)
            mu_x = mu_x + mhd_out

        hypotheses = self.final_layer(mu_x)

        return hypotheses, gate_idx

    def sample_predictor(self, coef_logits):
        '''
        Select predictor according to sampling policy.
        '''
        if self.mix_sampling_policy == 'greedy':
            # Predictor with highest coefficient is used for inference.
            predictor_idx = torch.argmax(coef_logits, dim=-1)

        elif self.mix_sampling_policy == 'multi':
            # According to paper
            predictor_idx = Categorical(logits=coef_logits / self.temperature).sample()

        elif self.mix_sampling_policy == 'random':
            # Vanilla WTA
            bsz = coef_logits.size(0)
            predictor_idx = torch.randint(0, self.mix_components, (bsz,))

        return predictor_idx
    
    def compute_loss(self, *args):
        if self.wta_loss == 'vanilla':
            return self._vanilla_wta_loss(*args)
        
        if self.wta_loss == 'mixture':
            return self._mixture_wta_loss(*args)
    
    def _vanilla_wta_loss(self, y,  hypotheses, *args):

        hypo_count = hypotheses.size(self.hypo_dim)
        y_shape = [1] * len(hypotheses.shape)
        y_shape[self.hypo_dim] = hypo_count

        # y: (batch size, hypo_count, out_dim)
        y_repeat = y.unsqueeze(self.hypo_dim).repeat(y_shape)
        
        distance_loss = self.dist_loss_fn(hypotheses, y_repeat).sum(-1)
        
        loss, pidx_best = torch.min(distance_loss, dim=-1)

        return loss.mean(-1), None, pidx_best

    def _mixture_wta_loss(self,  y, hypotheses, coef_logits):

        hypo_count = hypotheses.size(self.hypo_dim)
        y_shape = [1] * len(hypotheses.shape)
        y_shape[self.hypo_dim] = hypo_count

        # y: (batch size, hypo_count, out_dim)
        y_repeat = y.unsqueeze(self.hypo_dim).repeat(y_shape)
        
        distance_loss = self.dist_loss_fn(hypotheses, y_repeat).sum(-1)

        pidx_best = torch.argmin(distance_loss, dim=-1)
        # Create mask (w_i)
        loss_mask = F.one_hot(pidx_best, num_classes=hypo_count)

        log_coef = -coef_logits.log_softmax(-1)

        loss_dist  = (distance_loss * loss_mask).sum(-1)
        loss_coeff = (log_coef * loss_mask).sum(-1) * self.coef_weight

        return loss_dist, loss_coeff, pidx_best

    def init_weights(self):
        self.pre_layer.init_weights()
        self.final_layer.init_weights()

