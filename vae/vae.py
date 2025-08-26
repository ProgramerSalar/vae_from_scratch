import torch 
from torch import nn 
from typing import Tuple, Union
from dataclasses import dataclass


from timm.layers.weight_init import trunc_normal_
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils import BaseOutput

from .enc_dec import VaeEncoder, VaeDecoder, DiagonalGaussianDistribution
from .conv import vae_Conv3d


@dataclass 
class DecoderOutput(BaseOutput):

    """ 
    Output of decoding method.
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class VAE(ModelMixin, ConfigMixin):


    _supports_gradient_checkpointing = True


    @register_to_config
    def __init__(self,
                 #encoder realated parameters 
                 encoder_in_channels: int = 3,
                 encoder_out_channels: int = 4,
                 encoder_layer_per_block: Tuple[int, ...] = (2, 2, 2, 2),
                 encoder_down_block_types: Tuple[str, ...] = (
                     "DownEncoderBlockCausal3D",
                     "DownEncoderBlockCausal3D",
                     "DownEncoderBlockCausal3D",
                     "DownEncoderBlockCausal3D",
                 ),
                 encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 encoder_spatial_down_sample: Tuple[bool, ...] = (True, True, True, False),
                 encoder_temporal_down_sample: Tuple[bool, ...] = (True, True, True, False),
                 encoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
                 encoder_act_fn: str =  "silu",
                 encoder_norm_num_groups: int = 32,
                 encoder_double_z: bool = True,
                 encoder_type: str = "causal_vae_conv",
                 # decoder config 
                 decoder_in_channels: int = 4,
                 decoder_out_channels: int = 3,
                 decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
                 decoder_up_block_types: Tuple[str, ...] = (
                     "UpDecoderBlockCausal3D",
                     "UpDecoderBlockCausal3D",
                     "UpDecoderBlockCausal3D",
                     "UpDecoderBlockCausal3D",
                 ),
                 decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 decoder_spatial_up_sample: Tuple[bool, ...] = (True, True, True, False),
                 decoder_temporal_up_sample: Tuple[bool, ...] = (True, True, True, False),
                 decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
                 decoder_act_fn: str = "silu",
                 decoder_norm_num_groups: int = 32,
                 decoder_type: str = "causal_vae_conv",
                 # default values 
                 sample_size: int = 256,
                 scaling_factor: float = 0.18215,
                 add_post_quant_conv: bool = True,
                 interpolate: bool = False,
                 downsample_scale: int = 8
                 ):
        
        super().__init__()
        
        self.encoder = VaeEncoder(in_channels=encoder_in_channels,
                                  out_channels=encoder_out_channels,
                                  down_block_types=encoder_down_block_types,
                                  spatial_down_sample=encoder_spatial_down_sample,
                                  temporal_down_sample=encoder_temporal_down_sample,
                                  block_out_channels=encoder_block_out_channels,
                                  layers_per_block=encoder_layer_per_block,
                                  norm_num_groups=encoder_norm_num_groups,
                                  act_fn=encoder_act_fn,
                                  double_z=encoder_double_z,
                                  block_dropout=encoder_block_dropout,
                                  )
        

        self.decoder = VaeDecoder(in_channels=decoder_in_channels,
                                  out_channels=decoder_out_channels,
                                  up_block_types=decoder_up_block_types,
                                  spatial_up_sample=decoder_spatial_up_sample,
                                  temporal_up_sample=decoder_temporal_up_sample,
                                  block_out_channels=decoder_block_out_channels,
                                  layers_per_block=decoder_layers_per_block,
                                  norm_num_groups=decoder_norm_num_groups,
                                  interpolate=interpolate,
                                  block_dropout=decoder_block_dropout,
                                  act_fn=decoder_act_fn
                                  )
        

        self.quant_conv = vae_Conv3d(in_channels=2*encoder_out_channels,
                                     out_channel=2*encoder_out_channels,
                                     kernel_size=1,
                                     stride=1)
        self.post_quant_conv = vae_Conv3d(in_channels=encoder_out_channels,
                                          out_channel=encoder_out_channels,
                                          kernel_size=1,
                                          stride=1)
        

        # Tiling refers to splitting a large input (such as a video or image) into smaller pieces (tiles). 
        # These tiles are then processed one at a time (possibly with some overlap and blending), 
        # which helps reduce memory usage and makes it possible to process very large inputs that would not fit into memory all at once.
        self.use_tiling = False


        # only relevent if vae tiling is enabled 
        self.tile_sample_min_size = self.config.sample_size

        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )

        self.tile_latent_min_size = int(sample_size / downsample_scale) # 256 / 8 
        self.encode_tile_overlap_factor = 1 / 4 
        self.decode_tile_overlap_factor = 1 / 4 
        self.downsample_scale = downsample_scale 

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    

    def encode(self,
               x: torch.FloatTensor,
               return_dict: bool = True,
               is_init_image=True,
               temporal_chunk=False,
               window_size=16,
               tile_sample_min_size=256,
               ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        

        self.tile_latent_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)

        if self.use_tiling \
            and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):

            pass 

        if temporal_chunk:
            pass 

        else:
            h = self.encoder(x,
                             is_init_image=is_init_image,
                             temporal_chunk=False)
            
            moments = self.quant_conv(h,
                                      is_init_image=is_init_image,
                                      temporal_chunk=False)
            
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)
        
        return AutoencoderKLOutput(latent_dist=posterior)
    

    def decode(self,
               z: torch.FloatTensor,
               is_init_image=True,
               temporal_chunk=False,
               return_dict: bool = True,
               window_size: int = 2,
               tile_sample_min_size: int = 256) -> Union[DecoderOutput, torch.FloatTensor]:
        
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)

        if self.use_tiling \
            and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            
            pass 


        if temporal_chunk:
            pass 

        else:
            z = self.post_quant_conv(z,
                                     is_init_image=is_init_image,
                                     temporal_chunk=False)
            dec = self.decoder(z,
                               is_init_image=is_init_image,
                               temporal_chunk=False)
            

        if not return_dict:
            return_dict (dec,)

        return DecoderOutput(sample=dec)
    

    








    



        

        


        







if __name__ == "__main__":

    VAE.config_name = "vae_config"
    vae = VAE(encoder_norm_num_groups=1)
    print(vae)
 
    print("-" * 30)
    # x = torch.randn(2, 3, 8, 256, 256)
    # encoder = vae.encode(x=x)
    # print(encoder)

    z = torch.randn(2, 4, 1, 32, 32)
    decoder = vae.decode(z)
    print(decoder)