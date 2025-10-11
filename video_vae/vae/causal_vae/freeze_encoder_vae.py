import torch 
from torch import nn 
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from typing import List, Tuple, Union
from timm.layers.weight_init import trunc_normal_

from ..enc_dec import CausalEncoder, CausalDecoder, DecoderOutput
from ..conv import CausalConv3d
from ..gaussian import DiagonalGaussianDistribution
from middleware.gpu_processes import is_context_parallel_initialized, get_context_parallel_rank


class CausalVAE(ModelMixin, ConfigMixin):

    # this is the gradient checkpoint to reduce the memory used when you train the model.
    _supports_gradient_checkpointing = True

    # config_name = "CausalVAEConfig"

    @register_to_config
    def __init__(
            self,
            # <-- Encoder parameters --> 
            encoder_in_channels: int = 3,
            encoder_out_channels: int = 4,
            encoder_channels: List = [128, 256, 512, 512],
            down_num_layer: int = 2,
            encoder_num_layers: int = 4,
            encoder_dropout: float = 0.0,
            encoder_eps: float = 1e-6,
            encoder_norm_num_groups: int = 32,
            encoder_add_height_width_2x: Tuple[bool, ...] = (True, True, True, False),
            encoder_add_frame_2x: Tuple[bool, ...] = (True, True, True, False),
            encoder_double_z: bool = True,
            # <-- Decoder parameters --> 
            decoder_in_channels: int = 4,
            decoder_out_channels: int = 3,
            decoder_channels: List = [128, 256, 512, 512],
            up_num_layer: int = 3,
            decoder_num_layers: int = 4,
            decoder_dropout: float = 0.0,
            decoder_eps: float = 1e-6,
            decoder_norm_num_groups: int = 32,
            decoder_add_height_width_2x: Tuple[bool, ...] = (True, True, True, False),
            decoder_add_frame_2x: Tuple[bool, ...] = (True, True, True, False),
            scaling_factor: float = 1.0
    ):
        super().__init__()

        # [2, 3, 8, 256, 256] -> [2, 2*3, 1, 32, 32]
        self.encoder = CausalEncoder(in_channels=encoder_in_channels,
                                     out_channels=encoder_out_channels,
                                     channels=encoder_channels,
                                     down_num_layers=down_num_layer,
                                     encoder_num_layers=encoder_num_layers,
                                     dropout=encoder_dropout,
                                     eps=encoder_eps,
                                     scale_factor=scaling_factor,
                                     norm_num_groups=encoder_norm_num_groups,
                                     add_height_width_2x=encoder_add_height_width_2x,
                                     add_frame_2x=encoder_add_frame_2x,
                                     double_z=encoder_double_z)
        
        # [2, 2*3, 1, 32, 32] -> [2, 3, 1, 256, 256]
        self.decoder = CausalDecoder(in_channels=decoder_in_channels,
                                     out_channels=decoder_out_channels,
                                     channels=decoder_channels,
                                     up_num_layers=up_num_layer,
                                     decoder_num_layers=decoder_num_layers,
                                     dropout=decoder_dropout,
                                     eps=decoder_eps,
                                     scale_factor=scaling_factor,
                                     norm_num_groups=decoder_norm_num_groups,
                                     add_height_width_2x=decoder_add_height_width_2x,
                                     add_frame_2x=decoder_add_frame_2x,
                                    )
        
        # [2, 8, 1, 32, 32] -> [2, 8, 1, 32, 32]
        self.quant_conv = CausalConv3d(in_channels=2*encoder_out_channels,
                                       out_channels=2*encoder_out_channels,
                                       kernel_size=1,
                                       stride=1)
        
        # [2, 4, 1, 32, 32] -> [2, 4, 1, 32, 32]
        self.post_quant_conv = CausalConv3d(in_channels=encoder_out_channels,
                                            out_channels=encoder_out_channels,
                                            kernel_size=1,
                                            stride=1)
        
        self.use_tiling = False
        


    def _init_weight(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


   
    def forward(self,
                sample: torch.FloatTensor,
                sample_posterior: bool = True,
                generator: torch.Generator = None,
                freeze_encoder: bool = True
                ) -> Union[DecoderOutput, torch.FloatTensor]:
        

        x = sample

        if is_context_parallel_initialized():
            assert self.training, "You are training mode."

            if freeze_encoder:
                with torch.no_grad():
                    # [2, 3, 8, 256, 256] -> [2, 6, 1, 32, 32]
                    h = self.encoder(x)
                    # [2, 6, 1, 32, 32] -> [2, 6, 1, 32, 32]
                    moments = self.quant_conv(h)
                    posterior = DiagonalGaussianDistribution(moments)
                    global_posterior = posterior

            if sample_posterior:
                z = posterior.sample(generator=generator)

            if get_context_parallel_rank() == 0:
                dec = self.decode(z).shape

            else:
                dec = self.decode(z).shape


            return global_posterior, z
        



            


    def decode(self,
               z: torch.FloatTensor,
               return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        
        # [2, 4, 1, 32, 32] -> [2, 4, 1, 32, 32]
        z = self.post_quant_conv(z)

        # [2, 4, 1, 32, 32] -> [2, 3, 1, 256, 256]
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)
        
        return DecoderOutput(sample=dec)


   
        

        
            



    

            


    
     
        

        
        
        
        
        




        


