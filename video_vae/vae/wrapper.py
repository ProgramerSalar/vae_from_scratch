import torch 
from torch import nn 

import sys 
sys.path.append('../../vae_from_scratch/video_vae')
from vae.causal_vae import CausalVAE
from loss.loss import LossFunction


class CausalVideoLossWrapper(nn.Module):

    def __init__(self,
                 args,
                 num_groups):
        super().__init__()
        
        self.vae = CausalVAE(num_groups=num_groups)
        self.loss = LossFunction(perceptual_weight=args.perceptual_weight,
                                 pixelloss_weight=args.pixelloss_weight,
                                 logvar_init=args.logvar_init,
                                 kl_weight=args.kl_weight,
                                 disc_factor=args.disc_factor,
                                 disc_start=args.disc_start,
                                 disc_weight=args.disc_weight
                                 )

    
    def encode(self, x, sample=False):

        if sample:
            x = self.vae.encode(x).latent_dist.sample()
        else:
            x = self.vae.encode(x).latent_dist.mode()
        return x 
    
    def encode_latent(self, x, sample=False):

        latent = self.encode(x, sample=sample)
        return latent
    

    def forward(self, x, step):

        posterior, reconstruct = self.vae(x)

        # the reconstruct loss 
        reconstruct_loss, rec_log = self.loss(x,
                                              reconstruct,
                                              posterior,
                                              global_step=step,
                                              last_layer=self.vae.get_last_layer(),
                                              optimizer_idx=0)       
        gan_loss, gan_log  = self.loss(x,
                                    reconstruct,
                                    posterior,
                                    global_step=step,
                                    last_layer=self.vae.get_last_layer(),
                                    optimizer_idx=1)    

        loss_log = {**rec_log, **gan_log}
        
        
        return reconstruct_loss, gan_loss, loss_log   




if __name__ == "__main__":

    model = CausalVideoLossWrapper(num_groups=1)
    print(model)        

    x = torch.randn(1, 3, 8, 256, 256)
    # out = model.encode(x)
    encode_latent = model.encode_latent(x,sample=True)
    print(encode_latent)
