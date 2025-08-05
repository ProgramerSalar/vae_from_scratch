import torch 
from typing import Tuple, Dict, Union
from torch import nn 




# from diffusers.models.modeling_utils import ModelMixin
from .utils.model_mixin import ModelMixin

# from diffusers.configuration_utils import ConfigMixin, register_to_config
from .utils.config_mixin import ConfigMixin
from .utils.register_to_config import register_to_config
from .utils.utils import  trunc_normal_
from .modeling_enc_dec import (
    CausalVaeEncoder,
    CausalVaeDecoder,
    DiagonalGaussianDistribution
)
from .modeling_causal_conv import CausalConv3d


from diffusers.models.attention_processor import (
    AttentionProcessor,
    ADDED_KV_ATTENTION_PROCESSORS,
    AttnAddedKVProcessor,
    AttnProcessor,
    CROSS_ATTENTION_PROCESSORS
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput


class CausalVideoVAE(ModelMixin, 
                     ConfigMixin):
    
    r""" 
    
    A VAE model with KL loss for encoding images into latents and decoding latent representation into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented 
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int, *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.

        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional* to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.

        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-use standard deviation of the trained latent space computed using the first batch of the 
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the 
            diffusion model. When decoding the latents are scaled  back to the original scale with the formula: 
            `z = 1 / scaling_factor * z`. [if you know more: https://arxiv.org/abs/2112.10752] in this paper you 
            should lern section [4.3.2] [if you learn more details then goes to this papar: https://arxiv.org/pdf/2012.09841]

        force_upcast (`bool`, *optional* default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines. such as SD-XL. 
            VAE can be fine-tuned / trained to a lower range without loosing to much precision in which case 
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """


    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # encoder related parameters 
        encoder_in_channels: int = 3,
        encoder_out_channels: int = 4,
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
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
        encoder_act_fn: str = "silu",
        encoder_norm_num_groups: int = 32,
        encoder_double_z: bool = True,
        encoder_type: str = "causal_vae_conv", 
        # decoder related parameters 
        decoder_in_channels: int = 4,
        decoder_out_channels: int = 3,
        decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
        decoder_up_block_types: Tuple[str, ...] = (
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
        ),
        decoder_block_out_channels: Tuple[str, ...] = (128, 256, 512, 512),
        decoder_spatial_up_sample: Tuple[bool, ...] = (True, True, True, False),
        decoder_temporal_up_sample: Tuple[bool, ...] = (True, True, True, False),
        decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        decoder_act_fn: str = "silu",
        decoder_norm_num_groups: int = 32,
        decoder_type: str = "causal_vae_conv",
        sample_size: int = 256,
        scaling_factor: float = 0.18215,
        add_post_quant_conv: bool = True,
        interpolate: bool = False,
        downsample_scale: int = 8
    ):
        
    
        super().__init__()

        print(f"The latent dimension channels is {encoder_out_channels}")

        self.encoder = CausalVaeEncoder(
                in_channels=encoder_in_channels,
                out_channels=encoder_out_channels,
                down_block_types=encoder_down_block_types,
                spatial_down_sample=encoder_spatial_down_sample,
                temporal_down_sample=encoder_temporal_down_sample,
                block_out_channels=encoder_block_out_channels,
                layers_per_block=encoder_layers_per_block,
                norm_num_groups=encoder_norm_num_groups,
                act_fn=encoder_act_fn,
                double_z=True,
                block_dropout=encoder_block_dropout
        )

        # pass init param to Decoder 
        self.decoder = CausalVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            norm_num_groups=decoder_norm_num_groups,
            act_fn=decoder_act_fn,
            spatial_up_sample=decoder_spatial_up_sample,
            temporal_up_sample=decoder_temporal_up_sample,
            block_dropout=decoder_block_dropout,
            interpolate=interpolate
        )

        self.quant_conv = CausalConv3d(in_channels=2 * encoder_out_channels,
                                       out_channels= 2 * encoder_out_channels,
                                       kernel_size=1,
                                       stride=1)
        self.post_quant_conv = CausalConv3d(in_channels=encoder_out_channels,
                                            out_channels=encoder_out_channels,
                                            kernel_size=1,
                                            stride=1)
        
        self.use_tiling = False 

        # only relevent if vae tiling is enabled 
        self.tile_sample_min_size = self.config.sample_size 

        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )

        self.tile_latent_min_size = int(sample_size / downsample_scale)
        self.encoder_tile_overlap_factor = 1 / 4 
        self.decoder_tile_overlap_factor = 1 / 4 
        self.downsample_scale = downsample_scale

        self.apply(self.__init__weights)


    def __init__weights(self, m):

        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (self.encoder, self.decoder)):
            module.gradient_checkpointing = value 


    def enable_tiling(self, use_tiling: bool = True):

        r""" 
        Enable tiled VAE decoding. when this option is enabled. the vae will split the input tensor into tiles to 
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow 
        processing large images.
        """

        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enbale_tiling` was previously enabled. this method will go back to computing 
        decoding in one step
        """
        self.enable_tiling(False)


    @property
    #  Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:

        r""" 
        Returns:
            `dict` of attention processor: A dictionary containing all attention processors used in the model with 
            indexed by its weight name.
        """

        # set recursively 
        processors = {}

        def fn_recursive_and_processors(name: str,
                                        module: torch.nn.Module,
                                        processors: Dict[str, AttentionProcessor]):
            
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_and_processors(name=f"{name}.{sub_name}", 
                                            module=child,
                                            processors=processors)
                
            return processors
        

        for name, module in self.named_children():
            fn_recursive_and_processors(name=name,
                                        module=module,
                                        processors=processors)
            
        return processors
    
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self,
                           processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        

        r""" 
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` or `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                if `processor` is a dict, the key needs to define the path to the corresponding cross attention 
                processor. This is strongly recommended when setting trainable attention processors.
        
        """

        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processor {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )
    

        def fn_recursive_attn_processor(name: str,
                                        module: torch.nn.Module,
                                        processor):
            
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)

                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(name=f"{name}.{sub_name}",
                                            module=child,
                                            processor=processor)
                
        for name, module in self.named_children():
            fn_recursive_attn_processor(name=name,
                                        module=module,
                                        processor=processor)
            

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):

        r""" 
        Disables custom attention processor and sets the default attention implementation.
        """

        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()

        elif (proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()

        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        
        self.set_attn_processor(processor)

    
    def encode(self,
               x: torch.FloatTensor,
               return_dict: bool = True,
               is_init_image=True,
               temporal_chunk=False,
               window_size=16,
               tile_sample_min_size=256
               ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict: (`bool`, *optional*, defaults to `True`):
                whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            The latent representation of the encoded images. If `return_dict` is True, 
            a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plan `tuple` is returned.
        """

        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)

        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tile_encode()
        

    def tile_encode():
        pass 








