import torch 
from torch import nn 
from einops import rearrange

from vae.causal_vae.freeze_encoder_vae import CausalVAE
from wrapper_middleware.loss import LPIPSWithDiscriminator
from middleware.gpu_processes import is_context_parallel_initialized, get_context_parallel_world_size, get_context_parallel_group_rank, get_context_parallel_group

class CausalVideoVAELossWrapper(nn.Module):

    def __init__(self,
                 freeze_encoder: bool = True,
                 add_discriminator: bool = True,
                 load_loss_module: bool = True,
                 # <-- loss parameters --> 
                 disc_start=0,
                 logvar_init=0.0,
                 kl_weight=1e-6,
                 pixelloss_weight=1.0,
                 perceptual_weight=1.0,
                 disc_weight=0.1,
                 lpips_ckpt='vae/vgg_lpips.pth'
                 ):
        super().__init__()
        self.disc_start=disc_start
    

        self.vae = CausalVAE()
        self.vae_scale_factor = self.vae.config.scaling_factor 

        if freeze_encoder:
            for parameter in self.vae.encoder.parameters():
                parameter.requires_grad = False 

            for parameter in self.vae.quant_conv.parameters():
                parameter.requires_grad = False

        self.add_discriminator = add_discriminator
        self.freeze_encoder = freeze_encoder

        if load_loss_module:
            self.loss = LPIPSWithDiscriminator(disc_start=disc_start,
                                               logvar_init=logvar_init,
                                               kl_weight=kl_weight,
                                               pixelloss_weight=pixelloss_weight,
                                               perceptual_weight=perceptual_weight,
                                               lpips_ckpt=lpips_ckpt,
                                               disc_num_layers=4,
                                               disc_in_channels=3,
                                               disc_factor=1,
                                               disc_weight=disc_weight,
                                               using_3d_discriminator=False,
                                               add_discriminator=True,
                                               disc_loss="hinge")


            

        



    def forward(self, x, step):
        
        xdim = x.dim 
        if xdim == 4:
            x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, w)

        
        # video data 
        x = rearrange(x,
                      'b c t h w -> (b t) c h w')
        x = x.unsqueeze(2) # [(b t) c 1 h w]

        
        posterior, reconstruct = self.vae(sample=x,
                                          generator=torch.Generator().manual_seed(42),
                                          freeze_encoder=self.freeze_encoder)
        
        # The reconstruct loss 
        reconstruct_loss, rec_log = self.loss(
            inputs=x,                               # [16, 3, 1, 256, 256]
            reconstructions=reconstruct,            # [16, 4, 1, 256, 256]
            posteriors=posterior,
            optimizer_idx=0,
            global_step=step,
            last_layer=self.vae.get_last_layer()
        )

        if step < self.disc_start:
            return reconstruct_loss, None, rec_log
        
        # The loss to train the discriminator 
        gan_loss, gan_log = self.loss(inputs=x,
                                      reconstructions=reconstruct,
                                      posteriors=posterior,
                                      optimizer_idx=1,
                                      global_step=step,
                                      last_layer=self.vae.get_last_layer())
        
        loss_log = {**rec_log, **gan_log}
        
        return reconstruct_loss, gan_loss, loss_log
        
        
            
            









    

