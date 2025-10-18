import torch 
from torch import nn 
from causal_vae import CausalVAE
import sys 
from einops import rearrange

sys.path.append("/home/manish/Desktop/projects/vae_from_scratch/vae_from_scratch/video_vae")
from loss.lpips import Lpips


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 8, 256, 256).to(device)
    target = torch.randn(2, 3, 8, 256, 256).to(device)
    target = rearrange(target, 'b c t h w -> (b t) c h w')

    model = CausalVAE(device=device).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler(device=device)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = Lpips().to(device)
    # print(loss_fn)

    torch.autograd.set_detect_anomaly(True)
    
    # print(optimizer)

    for epoch in range(10):
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(x)
            output = rearrange(output, 'b c t h w -> (b t) c h w')
            loss = loss_fn(output, target)
            loss = loss.mean()
            print(f"Loss: {loss.item()}")

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
