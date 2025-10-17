import torch 
from torch import nn 


import torch 
from torch import nn 


class MyModel(nn.Module):

    def __init__(self,
                 in_channels: int,
                    out_channels: int):
        super().__init__()

        self.layer1 = nn.Linear(in_channels, out_channels)

        self.linear_layer = nn.Sequential(
            nn.Linear(in_channels, 2*out_channels),
            nn.Linear(2*out_channels, 3*out_channels),
            nn.Linear(3*out_channels, 4*out_channels),
            nn.Linear(4*out_channels, 5*out_channels),
            nn.Linear(5*out_channels, 6*out_channels),
            nn.Linear(6*out_channels, 7*out_channels),
            nn.Linear(7*out_channels, 8*out_channels),
            nn.Linear(8*out_channels,9*out_channels),
            nn.Linear(9*out_channels,10*out_channels),
            nn.Linear(10*out_channels,10*out_channels),
        )
        

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)

        torch.utils.checkpoint.checkpoint(
            self.linear_layer,
            x,
            use_reentrant=True
        )
        # x = self.linear_layer(x)
        x = self.relu(x)
        return x

    
model = MyModel(in_channels=100,
                out_channels=100)

from torch.utils.tensorboard import SummaryWriter 

writer = SummaryWriter('runs/testing')


data = torch.randn(100)
target = torch.randn(1000)
# output = model(data)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
scaler = torch.amp.GradScaler(device="cpu", enabled=True)

for epoch in range(100):

    optimizer.zero_grad()
    with torch.autocast(device_type="cpu", dtype=torch.float32, enabled=True):
        output = model(data)
        loss = loss_fn(output, target)
        print(loss.item())

    writer.add_scalar('Loss/train',loss.item(), epoch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

writer.close()
print("Training finished...")
