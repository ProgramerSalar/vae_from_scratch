import torch
from torch import nn 
from typing import List, Union, Tuple
from timm.models.layers import trunc_normal_
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from .enc_dec import CausalEncoder, CausalDecoder, DiagonalGaussianDistribution, DecoderOutput
from .conv import CausalConv3d
from .utils import (
    is_context_parallel_intialized
)


class CausalVideoVAE(ModelMixin, ConfigMixin):

    def __init__(self,
                 # encoder related parameters 
                 encoder_in_channels: int = 3,
                 encoder_out_channels: int = 4,
                 encoder_layer: int = 2,
                 encoder_num_layers: int = 4,
                 encoder_channels: List = [128, 256, 512, 512],
                 encoder_add_heigth_width_2x = (True, True, True, False),
                 encoder_add_frame_2x = (True, True, True, False),
                 encoder_double_z: bool = True,
                 # decoder related parameters 
                 decoder_in_channels: int = 4,
                 decoder_out_channels: int = 3,
                 decoder_channels: List = [128, 256, 512, 512],
                 decoder_layer: int = 3,
                 decoder_num_layers: int = 4,
                 decoder_add_height_width_2x=(True, True, True, False),
                 decoder_add_frame_2x = (True, True, True, False),
                 # default both parameters 
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 scale_factor: float = 0.18215,
                 norm_num_groups: int = 2
                 ):
        super().__init__()

        # [2, 3, 8, 256, 256] -> [2, 8, 1, 32, 32]
        self.encoder = CausalEncoder(
            in_channels=encoder_in_channels,
            out_channels=encoder_out_channels,
            channels=encoder_channels,
            num_layers=encoder_layer,
            encoder_num_layers=encoder_num_layers,
            dropout=dropout,
            scale_factor=scale_factor,
            norm_num_groups=norm_num_groups,
            add_height_width_2x=encoder_add_heigth_width_2x,
            add_frame_2x=encoder_add_frame_2x,
            double_z=encoder_double_z
        )

        # [2, 8, 1, 32, 32] -> [2, 8, 1, 32, 32]
        self.quant_conv = CausalConv3d(in_channels=2*encoder_out_channels,
                                       out_channels=2*encoder_out_channels,
                                       kernel_size=1,
                                       stride=1)
        
        # [2, 4, 1, 32, 32] -> [2, 4, 1, 32, 32]
        self.post_quant_conv = CausalConv3d(in_channels=encoder_out_channels,   # 4
                                            out_channels=encoder_out_channels,  # 4
                                            kernel_size=1,
                                            stride=1)
        

        # [2, 4, 1, 32, 32] -> [2, 3, 1, 256, 256]
        self.decoder = CausalDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            channels=decoder_channels,
            num_layers=decoder_layer,
            decoder_num_layers=decoder_num_layers,
            add_height_width_2x=decoder_add_height_width_2x,
            add_frame_2x=decoder_add_frame_2x,
            dropout=dropout,
            eps=eps,
            scale_factor=scale_factor,
            norm_num_groups=norm_num_groups
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def _set_gradient_checkpointing(self, 
                                    enable=True,
                                    gradient_checkpointing_func=None):
        
        if enable:
            def _apply_gradient_checkpointing(module):
                if isinstance(module, (CausalEncoder, CausalDecoder)):
                    module.gradient_checkpointing = True
            self.apply(_apply_gradient_checkpointing)

        else:
            def _apply_gradient_checkpointing(module):
                if isinstance(module, (CausalEncoder, CausalDecoder)):
                    module.gradient_checkpointing = False
            self.apply(_apply_gradient_checkpointing)



    def encode(
            self,
            x: torch.FloatTensor,
            return_dict: bool = True,
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        
        h = self.encode(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return_dict (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)
    

    def decode(self,
               z: torch.FloatTensor,
               return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)
        
        return DecoderOutput(sample=dec)
    
    
    def forward(self, 
                sample: torch.FloatTensor,
                sample_posterior: bool = True,
                generator: torch.Generator = None,
                freeze_encoder: bool = False,
                ) -> Union[DecoderOutput, torch.FloatTensor]:
        
        x = sample

        if is_context_parallel_intialized():
            pass

        else:
            # normal training perform 
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



    

        
        

        

    


        

        




    def forward(self, x):
        return x 
    



if __name__ == "__main__":

    causal_video_vae = CausalVideoVAE()
    print(causal_video_vae)