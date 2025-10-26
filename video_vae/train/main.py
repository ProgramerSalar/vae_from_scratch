import torch 
from torch import nn 
import sys 
from einops import rearrange
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append("../../vae_from_scratch/video_vae")
from loss.loss import LossFunction
from dataset.video_dataset import VideoDataset
from vae.causal_vae import CausalVAE



if __name__ == "__main__":

    
    # Instantiate the Dataset
    video_dataset = VideoDataset(video_dir='../../vae_from_scratch/Data/train_dataset')
    print(f"Dataset created with {len(video_dataset)} videos.")
    data_loader = DataLoader(video_dataset, batch_size=1, shuffle=True, num_workers=2)

   


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CausalVAE(num_groups=1).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    loss_fn = LossFunction().to(device)
    

    for epoch in range(10):
        optimizer.zero_grad()
        
        for batch in data_loader:
            print(f"epoch --> {epoch}")
            batch = batch.contiguous()
            print(f"train dataset shape: >>> {batch.shape}")

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                posterior, reconstruct = model(batch)
                print(posterior, reconstruct.shape)

                losses = loss_fn(batch, reconstruct, posterior, epoch, model.get_last_layer(), 0)
                print(f"losses  -> {losses}")
            

              
        

        