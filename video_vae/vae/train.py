import torch 
from torch import nn 

from causal_vae import CausalVAE


if __name__ == "__main__":

    x = torch.randn(2, 3, 8, 256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CausalVAE(device=device)
    output = model(x)
    optimizer = torch.optim.AdamW(params=model.parameters())
    scaler = torch.amp.GradScaler(device=device)
    
    # print(optimizer)

    for epoch in range(10):

        with torch.autocast(device_type=device, dtype=torch.float16):
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, x)
            print(f"Loss: {loss}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        