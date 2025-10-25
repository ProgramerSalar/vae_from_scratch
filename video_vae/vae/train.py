import torch 
from torch import nn 
from causal_vae import CausalVAE
import sys 
from einops import rearrange
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append("../../vae_from_scratch/video_vae")
from loss.loss import LossFunction
from dataset.video_dataset import VideoDataset



if __name__ == "__main__":

    ## Dataset 
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Instantiate the Dataset
    video_dataset = VideoDataset(video_dir='../../vae_from_scratch/Data/train_dataset', num_frames=16, transform=data_transform)
    print(f"Dataset created with {len(video_dataset)} videos.")
    data_loader = DataLoader(video_dataset, batch_size=2, shuffle=True, num_workers=2)

    # Test dataset 
    test_dataset = VideoDataset(video_dir='../../vae_from_scratch/Data/test_dataset', num_frames=16, transform=data_transform)
    print(f"Dataset created with {len(test_dataset)} videos.")
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=2)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x = torch.randn(2, 3, 8, 256, 256).to(device)
    # target = torch.randn(2, 3, 8, 256, 256).to(device)
    # target = rearrange(target, 'b c t h w -> (b t) c h w')

    model = CausalVAE(num_groups=1).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler(device=device)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = LossFunction()
    # print(loss_fn)

    torch.autograd.set_detect_anomaly(True)
    
    # print(optimizer)

    for epoch in range(10):
        optimizer.zero_grad()
        
        for batch in data_loader:
            batch = rearrange(batch, 'b t c h w -> b c t h w').to(device)
            batch = batch.contiguous()
            # print(f"train dataset shape: >>> {batch.shape}")

            output = model(batch)
            

              
        

            # scaler.scale(loss).backward()

            # # scaler.unscale_(optimizer)
            # # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step(optimizer)
            # scaler.update()