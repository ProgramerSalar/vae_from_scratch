import torch 
from torch import nn 

from causal_vae.freeze_encoder_vae import CausalVAE

class CausalVideoVAELossWrapper(nn.Module):

    def __init__(self):

        self.vae = CausalVAE()