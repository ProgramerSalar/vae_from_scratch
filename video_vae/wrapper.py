import torch
from torch import nn 
from typing import OrderedDict
from einops import rearrange

from vae import CausalVideoVAE
from loss import LPIPSWithDiscriminator

class CausalVideoVAELossWrapper(nn.Module):

    def __init__(self,
                 model_path=None,
                 model_dtype='fp32',
                 disc_start=0,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 perceptual_weight=1.0,
                 disc_weight=0.5,
                 interpolate=True,
                 add_discriminator=True,
                 freeze_encoder=False,
                 load_loss_module=False,
                 lpips_ckpt="vae/vgg_lpips.pth",
                 **kwargs):
        
        super().__init__()

        torch_dtype = torch.bfloat16 if model_dtype == "bf16" else torch.float32
        torch_dtype = torch.float16 if model_dtype == "fp16" else None

        # self.vae = CausalVideoVAE.from_pretrained(pretrained_model_name_or_path=model_path,
        #                                           torch_dtype=torch_dtype,
        #                                           interpolate=False)


        self.vae = CausalVideoVAE()

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.vae.encoder.parameters():
                parameter.requires_grad = False 
            for parameter in self.vae.quant_conv.parameters():
                parameter.requires_grad = False

        self.add_discriminator = add_discriminator

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
                                               disc_loss="hinge",
                                               add_discriminator=True,
                                               using_3d_discriminator=False
                                               )
        else:
            self.loss = None


        
    def load_checkpoint(self,
                        checkpoint_path,
                        **kwargs):
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        vae_checkpoint = OrderedDict()
        disc_checkpoint = OrderedDict()

        for key in checkpoint.keys():
            if key.startswith('vae.'):
                pass 
            if key.startswith('loss.discriminator'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[2:])
                disc_checkpoint[new_key] = checkpoint[key]

        vae_ckpt_load_result = self.vae.load_state_dict(vae_checkpoint, strict=False)
        if self.add_discriminator:
            disc_ckpt_load_result = self.loss.discriminator.load_state_dict(disc_checkpoint, strict=False)

    
    def forward(self, 
                x,
                step,
                identifier=['video']):
        
        xdim = x.ndim 
        if xdim == 4:
            x = x.unsqueeze(2)  # (b c h w) -> (b c 1 h w)
            
        if 'video' in identifier:
            assert 'image' not in identifier
        else:
            assert 'video' not in identifier
            x = rearrange(x,
                          'b c t h w -> (b t) c h w')
            x = x.unsqueeze(2) # [(b t) c 1 h w]

        batch_x = x 
        posterior, reconstruct = self.vae(sample=batch_x,
                                          freeze_encoder=self.freeze_encoder)
        
        print(posterior, reconstruct)
        
        # The reconstruction loss 
        # reconstruct_loss, rec_log = self.loss(inputs=batch_x,
        #                                       )
        


if __name__ == "__main__":
    out = CausalVideoVAELossWrapper()

    x = torch.randn(1, 3, 4, 256, 256)
    step = 1
    out = out(x, step)
    print(out)


        