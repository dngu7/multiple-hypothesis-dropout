import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from models.mlp import MLP
from models.mhdMLP import MHDropoutMLP
from models.utils import get_loss_fn, gather_on_indices

class MixtureOfMultiFunction(nn.Module):
    def __init__(self, 
        mix_components, 
        inp_dim, 
        hid_dim, 
        out_dim, 
        mhd_hid_dim, 
        num_layers=2, 
        act_fn='relu', 
        distance_loss='mse', 
        loss_aggregate='sum',
        sampling_policy='multi', 
        temperature=0.4, 
        random_restart=True, 
        restart_threshold=0.1,
        coef_weight=0.25, 
        out_act_fn=None, 
        wta_loss='vanilla',
        subset_ratio=0.5,
        gate_sampling='exact',
        var_weight=1.0):
        super().__init__()

        self.mix_layer = MLP(inp_dim, hid_dim, mix_components, 
                             num_layers=num_layers, act_fn=act_fn)

        self.ensemble = nn.ModuleList(
            [MHDropoutMLP(inp_dim, 
                hid_dim, 
                out_dim,
                mhd_hid_dim=mhd_hid_dim,
                num_layers=num_layers, 
                distance_loss=distance_loss,
                act_fn=act_fn, 
                out_act_fn=out_act_fn,
                temperature=temperature,
                coef_weight=coef_weight,
                wta_loss=wta_loss, 
                subset_ratio=subset_ratio,
                gate_sampling=gate_sampling,
                )\
              for _ in range(mix_components)])

        self.dist_loss_fn = get_loss_fn(distance_loss)(reduction='none')

        assert loss_aggregate in ['sum', 'mul'], loss_aggregate
        assert sampling_policy in ['greedy', 'multi', 'random'], sampling_policy

        self.mhd_hid_dim = mhd_hid_dim
        self.out_dim = out_dim
        self.loss_aggregate    = loss_aggregate
        self.mix_components    = mix_components
        self.sampling_policy   = sampling_policy
        self.random_restart    = random_restart
        self.temperature       = temperature
        self.coef_weight       = coef_weight
        
        # Store running usage of mixtures
        self.exp_usage = 1 / mix_components
        self.register_buffer('usage', torch.ones(mix_components) * self.exp_usage)

        self.restart_threshold = self.exp_usage * restart_threshold
        self.var_weight = var_weight


    def loss(self, x, y, *args, **kwargs):
        all_mu     = torch.stack([f.get_hypotheses(x, dropout=False)[0] for f in self.ensemble], dim=1)
        hypotheses = torch.stack([f.get_hypotheses(x, dropout=True)[0] for f in self.ensemble], dim=1)

        # Compute coefficients logits
        coef_logits = self.mix_layer(x)

        if self.training and self.random_restart:
            self.compute_usage(coef_logits) 
        
        #centre means
        loss_dist1, loss_coeff, pidx_best = self.compute_loss(y, all_mu, coef_logits)
        loss1 = loss_dist1 if loss_coeff is None else (loss_dist1 + loss_coeff * self.coef_weight)

        pred_best = gather_on_indices(pidx_best, hypotheses=hypotheses)

        loss_dist2, _ = self.compute_dist_loss(y, pred_best)

        loss = (loss1 + loss_dist2 * self.var_weight).mean()

        log_loss = {}
        log_loss['loss_dist1'] = loss_dist1.mean().detach().cpu().item()
        log_loss['loss_dist2'] = loss_dist2.mean().detach().cpu().item()
        if loss_coeff is not None:
            log_loss['loss_coeff'] = loss_coeff.mean().detach().cpu().item()
            
        return {'loss': loss, 'hypotheses': hypotheses}, log_loss #'pred_best': pred_best, 

    def sample(self, x, **kwargs):
        output_dict = {}
        
        hypo_count = 2 ** self.mhd_hid_dim - 1
        
        all_mu     = torch.stack([f.get_hypotheses(x, dropout=False)[0] for f in self.ensemble], dim=1)
        hypotheses = torch.stack([f.get_hypotheses(x, hypo_count=hypo_count, dropout=True)[0] for f in self.ensemble], dim=1)
        all_sigma  = hypotheses - all_mu.unsqueeze(2) 
        
        all_sigma_std  =  torch.std(all_sigma, dim=2) 
        
        coef_logits = self.mix_layer(x)
        pidx_sample = self.sample_predictor(coef_logits)

        mu = gather_on_indices(pidx_sample, hypotheses=all_mu)
        sigma = gather_on_indices(pidx_sample, hypotheses=all_sigma_std)
        samples = torch.randn_like(x) * sigma + mu

        output_dict['mu'] = mu
        output_dict['sigma'] = sigma
        output_dict['samples'] = samples

        return output_dict

    def gather_predictions(self, pidx, hypotheses):
        # Obtain chosen predictor's prediction
        if len(hypotheses.shape) == 3:
            arange0 = torch.arange(start=0, end=hypotheses.size(0), device=hypotheses.device)
            prediction = hypotheses[arange0,pidx,:]
        elif len(hypotheses.shape) == 4:
            expanded_indices = pidx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,self.out_dim)
            prediction = torch.gather(hypotheses, -2, expanded_indices).squeeze(-2)
        return prediction

    def sample_predictor(self, coef_logits):
        '''
        Select predictor according to sampling policy.
        '''
        if self.sampling_policy == 'greedy':
            # Predictor with highest coefficient is used for inference.
            predictor_idx = torch.argmax(coef_logits, dim=-1)

        elif self.sampling_policy == 'multi':
            # According to paper
            predictor_idx = Categorical(logits=coef_logits / self.temperature).sample()

        elif self.sampling_policy == 'random':
            # Vanilla WTA
            bsz = coef_logits.size(0)
            predictor_idx = torch.randint(0, self.mix_components, (bsz,))

        return predictor_idx

    def compute_dist_loss(self, y, hypotheses):

        # Prepare labels 
        if len(y.shape) == 2:
            components = hypotheses.size(1)
            repeat_frame = [1,components,1]
        else:
            raise ValueError("Unsupported shape")

        y_repeat = y.unsqueeze(-2).repeat(repeat_frame)

        # Calculate distance between each hypothesis and label
        distance_loss = self.dist_loss_fn(hypotheses, y_repeat).sum(-1)

        # Find best predictor
        best_pred_idx = torch.argmin(distance_loss, dim=-1)

        # Create mask (w_i)
        loss_mask = F.one_hot(best_pred_idx, num_classes=components)

        loss_dist  = (distance_loss * loss_mask).sum(-1)

        return loss_dist, best_pred_idx
        
    def compute_loss(self, y, hypotheses, coef_logits):

        # Prepare labels 
        if len(y.shape) == 2:
            components = hypotheses.size(1)
            repeat_frame = [1,components,1]
        else:
            raise ValueError("Unsupported shape")

        y_repeat = y.unsqueeze(-2).repeat(repeat_frame)

        # Calculate distance between each hypothesis and label
        distance_loss = self.dist_loss_fn(hypotheses, y_repeat).sum(-1)

        # Find best predictor
        best_pred_idx = torch.argmin(distance_loss, dim=-1)

        # Create mask (w_i)
        loss_mask = F.one_hot(best_pred_idx, num_classes=components)

        log_coef = -coef_logits.log_softmax(-1)

        loss_dist  = (distance_loss * loss_mask).sum(-1)
        loss_coeff = (log_coef * loss_mask).sum(-1)

        return loss_dist, loss_coeff, best_pred_idx
    
    def compute_usage(self, coef_logits, beta=0.25):
        '''
        Compute running average of coefficients
        '''
        with torch.no_grad():
            coefficients = coef_logits.softmax(-1)
            num_dims = len(coefficients.shape)
            dims = tuple(range(num_dims - 1))
            usage = torch.mean(coefficients, dim=dims)

            self.usage = beta * self.usage + (1. - beta) * usage  

    def restart_unused_mixtures(self):
        '''
        Resets predictors if usage falls below threshold.
        Must be executed outside of training loop to avoid inplace error.
        '''
        with torch.no_grad():
            restart_preds = torch.nonzero(torch.lt(self.usage, self.restart_threshold)).view(-1).tolist()
            if len(restart_preds) and self.random_restart:

                for pidx in restart_preds:
                    #Randomize weights
                    self.ensemble[pidx].init_weights()

                    #reset usage to expected usage
                    self.usage[pidx] = self.exp_usage

        

    
    