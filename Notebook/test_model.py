import torch
from torch import nn 


class Model(nn.Module):

    def __init__(self, 
                 in_channels,
                 out_channels):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=0),
        
            nn.GroupNorm(num_groups=2,
                                    num_channels=in_channels),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    ),
            nn.GroupNorm(num_groups=2,
                        num_channels=out_channels),
            nn.ReLU()
        
        )


    def forward(self, x):
        x = self.layer(x)
        print(f"what is the dtype: {x.dtype}")
        return x 
    


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(in_channels=128,
                out_channels=128).to(device)

    optim = torch.optim.Adam(params=model.parameters())
    loss_fn = nn.MSELoss()
    x = torch.randn(2, 128, 8, 256, 256, requires_grad=True).to(device)
    target = torch.randn(2, 128, 4, 252, 252, requires_grad=True).to(device)
    scaler = torch.GradScaler(device="cuda")

    for epoch in range(100):
        optim.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(x)
            loss = loss_fn(output, target)
            print(f"loss: {loss} and dtype: {loss.dtype}")
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()



    




