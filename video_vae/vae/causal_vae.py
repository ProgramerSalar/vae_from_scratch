import torch 
from torch import nn 
import sys 
sys.path.append('../../vae_from_scratch/video_vae')
from typing import Union
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from vae.enc import Encoder
from vae.conv import CausalConv3d
from vae.dec import Decoder
from vae.gaussian import DiagonalGaussianDistribution


class CausalVAE(nn.Module):

    def __init__(self,
                 num_groups,
                 encoder_out_channels=3,):

        super().__init__()
        
        self.encoder = Encoder(num_groups=num_groups)
        self.decoder = Decoder(num_groups=num_groups)

        self.quant_conv = CausalConv3d(in_channels=2*encoder_out_channels,
                                       out_channel=2*encoder_out_channels,
                                       kernel_size=3,
                                       stride=1)

        
    def get_last_layer(self):

        # [2, 3*2, 8, 256, 256]
        out = self.decoder.conv_out.conv.weight
        return out
    
    
    def forward(self, x, generator:torch.Generator=None):

        # <----------- Encoder part 
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        # <------------ Decoder part 
        z = posterior.sample(generator=generator)
        dec = self.decoder(z)

        return posterior, dec
        
    

    def encode(self, 
               x: torch.FloatTensor, 
               return_dict: bool = True) -> Union[AutoencoderKLOutput, tuple[DiagonalGaussianDistribution]]:
        
        h = self.encoder(x)
        moments = self.quant_conv(h)
        print(moments.shape)
        posterior = DiagonalGaussianDistribution(moments)

        return AutoencoderKLOutput(latent_dist=posterior)

    
    





if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalVAE(num_groups=2).to(device)
    print(model)

    learnable_param =  sum(param.numel() for param in model.parameters())
    print(f"learnable_parameters: {learnable_param / 1e6} Million")


    x = torch.randn(2, 3, 8, 256, 256).to(device)
    posterior, dec = model(x)
    print(posterior, dec.shape)

    # print('-'*40)
    # print(model.get_last_layer().shape)
    # -------------------------------------------------------------------

    # model = CausalVAE(num_groups=1)
    # print(model)

    # x = torch.randn(1, 3, 8, 256, 256)
    # # encode_latent = model.encode_latent(x)
    # # print(f"Encode Latents: {encode_latent}")
    # encode = model.encode(x)
    # print(f"Encode: {encode}")
    

