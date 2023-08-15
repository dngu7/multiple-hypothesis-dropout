import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_act_fn
from models.utils import get_loss_fn

class MLP(nn.Module):
    def __init__(self, 
                inp_dim, 
                hid_dim, 
                out_dim, 
                num_layers=2, 
                act_fn='relu', 
                distance_loss='mse', 
                out_act_fn=None, 
                dropout=0.0):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dist_loss_fn = get_loss_fn(distance_loss)()
        self.out_act_fn = get_act_fn(out_act_fn)() if out_act_fn is not None else None

        self.num_layers = num_layers
        assert num_layers > 0, "Must be more 0 layers"

        if num_layers == 1:
            self.pre_layer = nn.Linear(inp_dim, out_dim)
            self.hid_layer = None
            self.final_layer = None
        else:
            self.pre_layer = nn.Linear(inp_dim, hid_dim)
            self.final_layer = nn.Linear(hid_dim, out_dim)
            num_hid_layers = num_layers - 2
            hid_layer = [get_act_fn(act_fn)()]
            for _ in range(num_hid_layers):
                if dropout > 0.0:
                    hid_layer += [nn.Dropout(p=dropout, inplace=False)]
                hid_layer += [
                    nn.Linear(hid_dim, hid_dim),
                    get_act_fn(act_fn)()
                ]
            self.hid_layer = nn.Sequential(*hid_layer)
        
    def forward(self, x, y=None):
        x = self.pre_layer(x)
        if self.hid_layer is not None:
            x = self.hid_layer(x)
        if self.final_layer is not None:
            x = self.final_layer(x)
        if self.out_act_fn is not None:
            x = self.out_act_fn(x)
            
        outputs = (x, None, None, None,)
        if y is not None:
            loss = self.compute_loss(x, y)
            outputs += (loss,)
            return outputs
        else:
            return x

    def loss(self, x, y, *args, **kwargs):
        return self(x, y=y)[-1]

    def sample(self, x, *args, **kwargs):
        return {'pred_sample': self(x)}
    
    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def compute_loss(self, pred, target):
        return self.dist_loss_fn(pred, target)



