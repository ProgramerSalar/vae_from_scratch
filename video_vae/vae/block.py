import torch 
from torch import nn 

from .resnet import CausalResnet3d, DecreaseFeature, DecreaseFrame, IncreaseFeature, IncreaseFrame

class CausalDeownBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_groups,
                 enc_feature=False,
                 enc_frame=True,
                 output_scale_factor: int = 1.0,
                 resnet_embedding_norm: str = "default",
                 resnet_temb_channels: int = 512,
                 resnet_eps: float = 1e-6,
                 resnet_act_fn: str = "swish",
                 dropout: float = 0.0
                 ):
        
        super().__init__()
        self.enc_feature = enc_feature
        self.enc_frame = enc_frame

        self.down_layers = nn.ModuleList([])


        for i in range(2):

            input_channels = in_channels if i == 0 else out_channels
            # [128] -> [256]
            # [256] -> [256]
            self.down_layers.append(
                CausalResnet3d(in_channels=input_channels,
                               out_channels=out_channels,
                               num_groups=num_groups,
                               output_scale_factor=output_scale_factor,
                               time_embedding_norm=resnet_embedding_norm,
                               temb_channels=resnet_temb_channels,
                               eps=resnet_eps,
                               act_fn=resnet_act_fn,
                               dropout=dropout
                               )
            )

        
        
        if self.enc_feature is True:
            # <-- feature descrease --> 
            self.descrease_feature = DecreaseFeature(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    )
            
        
        
        if self.enc_frame is True:
            # <-- frame descrease --> 
            self.descrease_frame = DecreaseFrame(in_channels=out_channels,
                                                out_channels=out_channels,
                                                )
            
        
        

    def forward(self, x):
        
        for layer in self.down_layers:
            x = layer(x)


        # x = self.descrease_feature(x) if self.enc_feature is True else None
        # x = self.descrease_frame(x) if self.enc_frame is True else None

        if self.enc_feature:
            x = self.descrease_feature(x)

        if self.enc_frame:
            x = self.descrease_frame(x)



        return x 
    



class causalUpperBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_groups,
                 dec_feature=True,
                 dec_frame=True,
                 output_scale_factor: int = 1.0,
                 resnet_embedding_norm: str = "default",
                 resnet_temb_channels: int = 512,
                 resnet_eps: float = 1e-6,
                 resnet_act_fn: str = "swish",
                 dropout: float = 0.0):
        super().__init__()
        self.dec_feature = dec_feature
        self.dec_frame = dec_frame
        
        
        self.upper_block = nn.ModuleList([])
        for i in range(2):

            input_channels = in_channels if i == 0 else out_channels
            self.upper_block.append(
                CausalResnet3d(in_channels=input_channels,
                               out_channels=out_channels,
                               num_groups=num_groups,
                               output_scale_factor=output_scale_factor,
                               time_embedding_norm=resnet_embedding_norm,
                               temb_channels=resnet_temb_channels,
                               eps=resnet_eps,
                               act_fn=resnet_act_fn,
                               dropout=dropout
                               )
            )

        if self.dec_feature:
            # <-- Increase the feature --> 
            self.incr_feature = IncreaseFeature(in_channels=out_channels,
                                                out_channels=out_channels,
                                                )
        
        if self.dec_frame:
            # <-- Increment the frame --> 
            self.incr_frame = IncreaseFrame(in_channels=out_channels,
                                            out_channels=out_channels,
                                            )
        
        
    def forward(self, x):

        for layer in self.upper_block:
            x = layer(x)
        
        if self.dec_feature:
            x = self.incr_feature(x)
        
        if self.dec_frame:
            x = self.incr_frame(x)

        return x 





if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CausalDeownBlock(in_channels=128,
    #                          out_channels=256,
    #                          num_groups=2,
    #                          )
    # print(model)

    # # learnable_parameters = sum(parm.numel() for parm in model.parameters())
    # # print(f"learnable_parameters = {learnable_parameters / 1000000 :.3f} Million")


    # x = torch.randn(2, 128, 8, 256, 256)
    # out = model(x)
    # print(out.shape)
    # ---
    model = causalUpperBlock(in_channels=256,
                             out_channels=128,
                             num_groups=2)
    
    x = torch.randn(2, 256, 1, 16, 16)
    out = model(x)
    print(out.shape)    # [2, 128, 2, 32, 32]
        