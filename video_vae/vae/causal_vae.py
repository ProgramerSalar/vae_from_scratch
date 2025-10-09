import torch 
from torch import nn 
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from typing import List, Tuple, Union
from timm.layers.weight_init import trunc_normal_

from .enc_dec import CausalEncoder, CausalDecoder, DecoderOutput
from .conv import CausalConv3d
from .gaussian import DiagonalGaussianDistribution

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
        
        # [2, 8, 1, 256, 256] -> [2, 8, 1, 256, 256]
        self.quant_conv = CausalConv3d(in_channels=2*encoder_out_channels,
                                       out_channels=2*encoder_out_channels,
                                       kernel_size=1,
                                       stride=1)
        
        # [2, 4, 1, 256, 256] -> [2, 4, 1, 256, 256]
        self.post_quant_conv = CausalConv3d(in_channels=encoder_out_channels,
                                            out_channels=encoder_out_channels,
                                            kernel_size=1,
                                            stride=1)
        


    def _init_weight(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def _set_gradient_checkpointing(self, 
                                    enable = False, 
                                    gradient_checkpointing_func=None):
        
        if enable:
            def _apply_gradient_checkpointing(module):
                if isinstance(module, (CausalEncoder, CausalDecoder)):
                    module.gradient_checkpointing = True

            self.apply(_apply_gradient_checkpointing)

        else:
            raise NotImplementedError("You are not in the Training mode. Please you should activate the training mode!")
        
    ###  <-------------------------------- Tile function -------------------------------> ###

    def enable_tiling(self,
                      use_tiling: bool = True):
        
        """
            Enable tiled VAE decoding. when this option is enabled, the VAE will split the input tensor into tiles to
            compute decoding and encoding in several steps. 
            This is useful for saving a large amount of memory and to allow processing larger images.

            for more info: https://arxiv.org/html/2412.15185v4

            PURPOSE: 
                Tiled VAE Decoding Explained
                Imagine you have to paint a very large picture.

                Normal Method: You try to paint the entire canvas all at once. 
                This requires a huge workspace and all your paint tubes to be open, 
                which can be overwhelming and messy.

                Tiled Method: You divide the canvas into a grid of smaller squares (tiles). 
                You paint one square at a time, finish it, and then move to the next. 
                This requires a much smaller workspace and only the paints you need for that specific square.
        """
        
        self.use_tiling = use_tiling


    def disable_tiling(self):

        """
            Disable tiled VAE decoding. 
            If `enable_tiling` was previously enabled, this method will go back to computing 
            decoding in one step.
        """

        self.enable_tiling(use_tiling=False)

    ###  <-------------------------------- Tile function -------------------------------> ###


    def encode(
            self,
            x: torch.FloatTensor,
            return_dict: bool = True,
            window_size=16,
            tile_sample_min_size=256,
            temporal_chunk=False
    ) -> Union[Tuple[DiagonalGaussianDistribution], AutoencoderKLOutput]:
        
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_sample_min_size = int(tile_sample_min_size / 8)
        
        # [2, 3, 256, 256]
        if self.use_tiling and \
            (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size): # feature > 32

            assert NotImplementedError("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Your Image size is greater than 32.>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if temporal_chunk:
            """
                a type of generative model that compresses video data by dividing it into sequential "chunks" and processing them 
                with a Variational Autoencoder (VAE). This technique is used primarily in video generation and c
                ompression to efficiently handle long videos and maintain temporal consistency. 
            """
            
            assert NotImplementedError("make sure you don't provide the long videos in you dataset")


        # [2, 3, 32, 32]
        else:
            assert NotImplementedError("Your Image size is Lower than 32.")

            h = self.encoder(x)
            moments = self.uant_conv(h)




        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)
        
        return AutoencoderKLOutput(latent_dist=posterior)
    


    def decode(self,
               z: torch.FloatTensor,
               window_size=2,
               return_dict: bool = True,
               tile_sample_min_size=256,
               temporal_chunk=False) -> Union[torch.FloatTensor, DecoderOutput]:
        

        self.tile_sample_min_size = tile_sample_min_size
        self.tile_sample_min_size = int(tile_sample_min_size / 8)

        if self.use_tiling and \
            (z.shape[-1] > self.tile_sample_min_size or z.shape[-2] > self.tile_sample_min_size): # feature > 32

            assert NotImplementedError("Your Image size is greater than 32.")


        if temporal_chunk:
            
            """
                a type of generative model that compresses video data by dividing it into sequential "chunks" and processing them 
                with a Variational Autoencoder (VAE). This technique is used primarily in video generation and c
                ompression to efficiently handle long videos and maintain temporal consistency. 
            """
            
            assert NotImplementedError("make sure you don't provide the long videos in you dataset")

        else:
            print("<<<<<<<<<<<<<<Your Image size is Lower than 32.>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            z = self.post_quant_conv(z)
            dec = self.decoder(z)

        if not return_dict:
            return (dec,)
        
        return DecoderOutput(sample=dec)
        
            



    

            


    
     
        

        
        
        
        
        




        


