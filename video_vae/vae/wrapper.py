import torch 
from torch import nn 

import sys 
sys.path.append('../../vae_from_scratch/video_vae')
from vae.causal_vae import CausalVAE



class CausalVideoLossWrapper(nn.Module):

    def __init__(self,
                 num_groups):
        super().__init__()
        
        self.vae = CausalVAE(num_groups=num_groups)

    
    def encode(self, x, sample=False):

        if sample:
            x = self.vae.encode(x).latent_dist.sample()
        else:
            x = self.vae.encode(x).latent_dist.mode()
        return x 
    
    def encode_latent(self, x, sample=False):

        latent = self.encode(x, sample=sample)
        return latent
    

    def forward(self, x):
        pass 




if __name__ == "__main__":

    model = CausalVideoLossWrapper(num_groups=1)
    print(model)        

    x = torch.randn(1, 3, 8, 256, 256)
    # out = model.encode(x)
    encode_latent = model.encode_latent(x,sample=True)
    print(encode_latent)
