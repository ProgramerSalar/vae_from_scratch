import torch 
from torch import nn 
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Tuple, List, Union, Optional
from timm.models.layers import trunc_normal_
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from enc_dec import CausalEncoder, CausalDecoder, DiagonalGaussianDistribution, DecoderOutput
from conv import CausalConv3d

class CausalVideoVAE(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            # <--- encoder related parameters --->
            encoder_in_channels: int = 3,
            encoder_out_channels: int = 4,
            encoder_channels: List = [128, 256, 512, 512],
            num_layers: int = 2,
            encoder_num_layers: int = 4,
            encoder_dropout: float = 0.0,
            encoder_eps: float = 1e-6,
            encoder_scale_factor: float = 1.0,
            encoder_norm_num_groups: int = 32,
            encoder_add_height_width_2x: Tuple[bool, ...] = (True, True, True, False),
            encoder_add_frame_2x: Tuple[bool, ...] = (True, True, True, True, False),
            encoder_double_z: bool = True,
            # <--- Decoder related parameters ---> 
            decoder_in_channels: int = 4,
            decoder_out_channels: int = 3,
            decoder_channels: List = [128, 256, 512, 512],
            decoder_num_layers: int = 4,
            decoder_dropout: float = 0.0,
            decoder_eps: float = 1e-5,
            decoder_scale_factor: float = 1.0,
            decoder_norm_num_groups: int = 32,
            decoder_add_height_width_2x: Tuple[bool, ...] = (True, True, True, False),
            decoder_add_frame_2x: Tuple[bool, ...] = (True, True, True, False)
    ):
        super().__init__()
        
        # this is the gradient checkpoint to reduce the memory used when you train the model
        _supports_gradient_checkpointing = True

        self.encoder = CausalEncoder(in_channels=encoder_in_channels,
                                     out_channels=encoder_out_channels,
                                     channels=encoder_channels,
                                     num_layers=3,
                                     encoder_num_layers=encoder_num_layers,
                                     dropout=encoder_dropout,
                                     eps=encoder_eps,
                                     scale_factor=encoder_scale_factor,
                                     norm_num_groups=encoder_norm_num_groups,
                                     add_height_width_2x=encoder_add_height_width_2x,
                                     add_frame_2x=encoder_add_frame_2x,
                                     double_z=encoder_double_z
                                     )
        
        self.decoder = CausalDecoder(in_channels=decoder_in_channels,
                                     out_channels=decoder_out_channels,
                                     channels=decoder_channels,
                                     num_layers=2,
                                     decoder_num_layers=decoder_num_layers,
                                     dropout=decoder_dropout,
                                     eps=decoder_eps,
                                     scale_factor=decoder_scale_factor,
                                     norm_num_groups=decoder_norm_num_groups,
                                     add_height_width_2x=decoder_add_height_width_2x,
                                     add_frame_2x=decoder_add_frame_2x)
        

        self.quant_conv = CausalConv3d(in_channels=2*encoder_out_channels,
                                       out_channels=2*encoder_out_channels,
                                       kernel_size=1,
                                       stride=1)
        
        self.post_quant_conv = CausalConv3d(in_channels=encoder_out_channels,
                                            out_channels=encoder_out_channels,
                                            kernel_size=1,
                                            stride=1)
        

    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def encode(self,
               x: torch.FloatTensor,
               return_dict: bool = True
               ) -> Union[AutoencoderKLOutput, DiagonalGaussianDistribution]:
        

        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)
        
        return AutoencoderKLOutput(latent_dist=posterior)
    
    def decode(self,
               z: torch.FloatTensor,
               return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)
        
        return DecoderOutput(sample=dec)

        
    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight
        

    def forward(self,
                sample: torch.FloatTensor,
                sample_posterior: bool = True,
                generator: Optional[torch.Generator] = None,
                freeze_encoder: bool = False,
                ) -> Union[DecoderOutput, torch.FloatTensor]:
        
        x = sample 

        # The normal training 
        if freeze_encoder:
            with torch.no_grad():
                posterior = self.encode(x).latent_dist

        else:
            posterior = self.encode(x).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z).sample

        return posterior, dec



if __name__ == "__main__":
    out = CausalVideoVAE()
    x = torch.randn(1, 3, 4, 256, 256)
    out = out(x)
    print(out)