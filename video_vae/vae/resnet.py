import torch 
from torch import nn 
from einops import rearrange, reduce, repeat
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.activations import get_activation

from .conv import CausalConv3d, CausalGroupNorm

class CausalResnet3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int,
                 output_scale_factor:int=1.0,
                 time_embedding_norm: str = "default",
                 temb_channels: int = 512,
                 eps:float=1e-6,
                 act_fn:str="swish",
                 conv_shortcut_bias=True,
                 dropout: float = 0.0
                 ):
        
        """
        Args:
            time_embedding_norm: "default", "ada_groups", "spatial"
            act_fn: "gelu", "relu", "swish"
        """
        
        super().__init__()
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        
        # <---------- Normalization ---------------->
        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(embedding_dim=temb_channels,
                                      out_dim=in_channels,
                                      num_groups=num_groups,
                                      eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(f_channels=in_channels,
                                     zq_channels=temb_channels)
        else:
            self.norm1 = CausalGroupNorm(in_channels=in_channels,
                                        num_groups=num_groups,
                                        eps=eps,
                                        affine=True
                                        )

        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channel=out_channels,
                                  kernel_size=3,
                                  stride=1
                                  )
        
        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(embedding_dim=temb_channels,
                                      out_dim=out_channels,
                                      num_groups=num_groups,
                                      eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(f_channels=out_channels,
                                     zq_channels=temb_channels)
        else:
            self.norm2 = CausalGroupNorm(in_channels=out_channels,
                                     num_groups=num_groups,
                                     eps=eps,
                                     affine=True
                                     )
        output_channels = out_channels
        self.conv2 = CausalConv3d(in_channels=out_channels,
                                  out_channel=out_channels,
                                  kernel_size=3,
                                  stride=1
                                 )
        
        
        self.dropout = nn.Dropout(dropout)
        self.act_fn = get_activation(act_fn)

        # 128 -> 256
        # output_channels = out_channels
        self.increment_conv = None
        if in_channels != out_channels:
            self.increment_conv = CausalConv3d(in_channels=in_channels,
                                               out_channel=output_channels,
                                               kernel_size=3,
                                               stride=1,
                                               bias=conv_shortcut_bias)
        
    def forward(self, x, temb: torch.FloatTensor=None):
        t = x.shape[2]
        sample = x

        if self.time_embedding_norm == "spatial":
            temb = repeat(temb, 'b c -> (b t) c', t=t)
            temb = temb.unsqueeze(-1).unsqueeze(-1).contiguous() # [2*8, 128, 1, 1]
        

        if self.time_embedding_norm == "ada_group":
            x = self.norm1(x, temb)

        if self.time_embedding_norm == "spatial":
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.norm1(x, temb)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t).contiguous()

        else:
            x = self.norm1(x)

        x = self.act_fn(x)
        x = self.conv1(x)

        if temb is not None and self.time_embedding_norm == "default":
            x = x + temb.unsqueeze(1).unsqueeze(1).unsqueeze(1).contiguous()
        
        if self.time_embedding_norm == "ada_group":
            x = self.norm2(x)

        if self.time_embedding_norm == "spatial":
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.norm2(x, temb)
            temb = temb.squeeze(-1).squeeze(-1)
            temb = reduce(temb, '(b t) c -> b c', t=t, reduction="max").contiguous()
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t).contiguous()


        else:
            x = self.norm2(x)

        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.conv2(x)



        # [2, 128, 8, 256, 256]
        if self.increment_conv is not None:
            sample = self.increment_conv(sample)

        x = x + sample / self.output_scale_factor

        return x 
    

class DecreaseFeature(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__()
        

        stride = (1, 2, 2)
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channel=out_channels,
                                 stride=stride)
        
    def forward(self, x):

        x = self.conv(x)
        return x 
        

        
class DecreaseFrame(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__()
        

        stride = (2, 1, 1)
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channel=out_channels,
                                 stride=stride)
        
    def forward(self, x):

        x = self.conv(x)
        return x 



class IncreaseFeature(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        
        super().__init__()

        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channel=out_channels * 4,
                                 kernel_size=3,
                                 stride=1)

        
    def forward(self, x):

        # [2, 512, 1, 16, 16] -> [2, 512*4, 1, 16, 16]
        x = self.conv(x)

        x = rearrange(x, 
                      'b (c p1 p2) t h w -> b c t (h p1) (w p2)', p1=2, p2=2).contiguous()
        
        return x 
    

class IncreaseFrame(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        
        super().__init__()
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channel=out_channels * 2,
                                 kernel_size=3,
                                 )
        

    def forward(self, x):

        t = x.shape[2]
        x = self.conv(x)
        x = rearrange(x,
                      'b (c p) t h w -> b c (t p) h w', t=t, p=2).contiguous()
        
        return x 



if __name__ == "__main__":
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalResnet3d(in_channels=128,
                           out_channels=256,
                           num_groups=2,
                           time_embedding_norm="default",
                           temb_channels=256
                           )
    
    print(model)
    x = torch.randn(2, 128, 8, 256, 256)
    t = torch.randn(2, 256)
    out = model(x, t)
    print(out.shape)
    # ---
    # model = DecreaseFrame(in_channels=128,
    #                         out_channels=128,
    #                         device=device)
    
    # x = torch.randn(2, 128, 8, 256, 256)
    # out = model(x)
    # print(out.shape)
    # -----

    # model = IncreaseFeature(in_channels=512, 
    #                         out_channels=512,
    #                         device=device)
    # x = torch.randn(2, 512, 1, 16, 16)
    # out = model(x)
    # print(out.shape)
    # ---

    # model = IncreaseFrame(in_channels=512, 
    #                       out_channels=512,
    #                       )
    
    # x = torch.randn(2, 512, 1, 16, 16)
    # out = model(x)
    # print(out.shape)
        
