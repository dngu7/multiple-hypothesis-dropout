import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from models.mlp import MLP
from models.utils import get_loss_fn

class MixtureFFN(nn.Module):
    '''
    Mixture of feed forward networks with standard dropout.
    '''
    def __init__(self, mix_components, inp_dim, hid_dim, out_dim, 
        num_layers=2, act_fn='relu', distance_loss='mse', loss_aggregate='sum',
        sampling_policy='multi', temperature=0.4, random_restart=True, 
        restart_threshold=0.1, coef_weight=0.25, out_act_fn=None, dropout=0.0):
        super().__init__()

        self.mix_layer = MLP(inp_dim, hid_dim, mix_components, 
                             num_layers=num_layers, act_fn=act_fn)

        self.ensemble  = nn.ModuleList(
            [MLP(inp_dim, 
                hid_dim, 
                out_dim, 
                num_layers=num_layers, 
                act_fn=act_fn, 
                dropout=dropout,
                out_act_fn=out_act_fn)\
              for _ in range(mix_components)])

        self.dist_loss_fn = get_loss_fn(distance_loss)(reduction='none')

        assert loss_aggregate in ['sum', 'mul'], loss_aggregate
        assert sampling_policy in ['greedy', 'multi', 'random'], sampling_policy

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

    def loss(self, x, y, *args, **kwargs):
        return self(x, y=y, *args, **kwargs)

    def sample(self, x, **kwargs):
        out, _ =  self(x)
        return out
    
    def forward(self, x, y=None, *args, **kwargs):
        output_dict = {}

        # Obtain hypothesis from each predictor
        hypotheses = torch.stack([f(x) for f in self.ensemble], dim=-2)

        # Compute coefficients logits
        coef_logits = self.mix_layer(x)

        if self.training and self.random_restart:
            self.compute_usage(coef_logits) 

        # Select predictor from ensemble.
        pidx_sample = self.sample_predictor(coef_logits)
        pred_sample = self.gather_predictions(pidx_sample, hypotheses=hypotheses)

        output_dict['pred_sample'] = pred_sample
        output_dict['pidx_sample'] = pidx_sample
        output_dict['hypotheses']  = hypotheses
        output_dict['coef_logits'] = coef_logits

        log_loss = {}
        if y is not None:
            loss_dist, loss_coeff, pidx_best = self.compute_loss(y, hypotheses, coef_logits)

            log_loss['loss_dist']  = loss_dist.mean().detach().cpu().item()
            log_loss['loss_coeff'] = None if loss_coeff is None else loss_coeff.mean().detach().cpu().item()

            output_dict['loss']  = (loss_dist + loss_coeff * self.coef_weight).mean(-1)

            pred_best = self.gather_predictions(pidx_best, hypotheses=hypotheses)
            output_dict['pidx_best'] = pidx_best
            output_dict['pred_best'] = pred_best
        
        return output_dict, log_loss

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
    
    def compute_loss(self, y, hypotheses, coef_logits):
        '''
        Mixture-WTA Loss: Combines least-squares and maximum likelihood.
        
        Not suitable for sequences. see Sketchmcl.py
        '''
        # Prepare labels 
        if len(y.shape) == 2:
            repeat_frame = [1,self.mix_components,1]
        else:
            raise ValueError("Unsupported shape")


        y_repeat = y.unsqueeze(-2).repeat(repeat_frame)
        
        # Calculate distance between each hypothesis and label
        distance_loss = self.dist_loss_fn(hypotheses, y_repeat).sum(-1)

        # Find best predictor
        best_pred_idx = torch.argmin(distance_loss, dim=-1)

        # Create mask (w_i)
        loss_mask = F.one_hot(best_pred_idx, num_classes=self.mix_components)

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

        

    
    