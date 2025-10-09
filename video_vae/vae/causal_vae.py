import torch 
from torch import nn 
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import List, Tuple

from .enc_dec import CausalEncoder, CausalDecoder
from .conv import CausalConv3d

class CausalVAE(ConfigMixin, ModelMixin):

    @register_to_config
    def __init__(
            self,
            # <-- Encoder parameters --> 
            encoder_in_channels: int = 3,
            encoder_out_channels: int = 4,
            encoder_channels: List[int, int, int, int] = [128, 256, 512, 512],
            down_num_layer: int = 2,
            encoder_num_layers: int = 4,
            encoder_dropout: float = 0.0,
            encoder_eps: float = 1e-6,
            encoder_scale_factor: float = 1.0,
            encoder_norm_num_groups: int = 32,
            encoder_add_height_width_2x: Tuple[bool, ...] = (True, True, True, False),
            encoder_add_frame_2x: Tuple[bool, ...] = (True, True, True, False),
            encoder_double_z: bool = True,
            # <-- Decoder parameters --> 
            decoder_in_channels: int = 4,
            decoder_out_channels: int = 3,
            decoder_channels: List[int, int, int, int] = [128, 256, 5121, 512],
            up_num_layer: int = 3,
            decoder_num_layers: int = 4,
            decoder_dropout: float = 0.0,
            decoder_eps: float = 1e-6,
            decoder_scale_factor: float = 1.0,
            decoder_norm_num_groups: int = 32,
            decoder_add_height_width_2x: Tuple[bool, ...] = (True, True, True, False),
            decoder_add_frame_2x: Tuple[bool, ...] = (True, True, True, False)
    ):
        super().__init__()

        # this is the gradient checkpoint to reduce the memory used when you train the model.
        _supports_gradient_checkpointing = True

        self.encoder = CausalEncoder(in_channels=encoder_in_channels,
                                     out_channels=encoder_out_channels,
                                     channels=encoder_channels,
                                     down_num_layers=down_num_layer,
                                     encoder_num_layers=encoder_num_layers,
                                     dropout=encoder_dropout,
                                     eps=encoder_eps,
                                     scale_factor=encoder_scale_factor,
                                     norm_num_groups=encoder_norm_num_groups,
                                     add_height_width_2x=encoder_add_height_width_2x,
                                     add_frame_2x=encoder_add_frame_2x,
                                     double_z=encoder_double_z)
        
        self.decoder = CausalDecoder(in_channels=decoder_in_channels,
                                     out_channels=decoder_out_channels,
                                     channels=decoder_channels,
                                     up_num_layers=up_num_layer,
                                     decoder_num_layers=decoder_num_layers,
                                     dropout=decoder_dropout,
                                     eps=decoder_eps,
                                     scale_factor=decoder_scale_factor,
                                     norm_num_groups=decoder_norm_num_groups,
                                     add_height_width_2x=decoder_add_height_width_2x,
                                     add_frame_2x=decoder_add_frame_2x,
                                    )
        

        self.quant_conv = CausalConv3d(in_channels=2*encoder_out_channels,
                                       out_channels=2*encoder_out_channels,
                                       kernel_size=1,
                                       stride=1)
        
        
        




        


