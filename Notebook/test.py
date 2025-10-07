import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 ):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=2)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # assert self.training, "make sure training mode is activate"
        assert not self.training, "make sure training mode is  not activate"

        x = self.norm(x)
        x = self.conv(x)
        x = self.norm2(x)
        print(x.shape)

        return x 
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    out = SimpleModel(in_channels=3,
                    out_channels=3)
    # out.train(mode=True)
    out = out(x)
    out.shape
