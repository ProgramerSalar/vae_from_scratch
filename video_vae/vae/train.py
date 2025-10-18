import torch 
from torch import nn 
from causal_vae import CausalVAE
import sys 
sys.path.append("/home/manish/Desktop/projects/vae_from_scratch/vae_from_scratch/video_vae")
from loss.lpips import Lpips


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 8, 256, 256).to(device)
    target = torch.randn(2, 3, 8, 256, 256).to(device)

    model = CausalVAE(device=device).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters())
    scaler = torch.amp.GradScaler(device=device)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = Lpips()
    print(loss_fn)
    
    # print(optimizer)

    for epoch in range(10):
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(x)
            loss = loss_fn(output, target)
            print(f"Loss: {loss}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        