import torch 
from torch import nn 

from diffusers.models.attention_processor import Attention

class AttentionLayer(nn.Module):

    def __init__(self,
                 in_channels=512,
                 num_groups=2,
                 ):
        super().__init__()

        self.attention_layer = Attention(query_dim=in_channels,
                                         dim_head=512,
                                         heads=1,
                                         norm_num_groups=num_groups,
                                         residual_connection=True,
                                         bias=True,
                                         upcast_softmax=True,
                                         _from_deprecated_attn_block=True,
                                         rescale_output_factor=1.0,
                                         eps=1e-5,
                                         )

    def forward(self, x):
        
        x = self.attention_layer(x)
        return x 
    

if __name__ == "__main__":

    model = AttentionLayer()
    print(model)

    n_learnable_param = sum(param.numel() for param in model.parameters())
    print(f"Learnable parametes : {n_learnable_param / 1e6} Million")

    x = torch.randn(2, 512, 32, 32)
    out = model(x)
    print(out.shape)