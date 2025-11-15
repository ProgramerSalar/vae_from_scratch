import torch 
from torch.utils.data import DataLoader
import sys 
sys.path.append("../../vae_from_scratch/video_vae")
from dataset.video_dataset import VideoDataset


def video_dataloader(args):

  dir = "../../vae_from_scratch/video_vae/dataset/video"
  ds = VideoDataset(dir)
  loader = DataLoader(dataset=ds,
                      batch_size=args.batch_size,
                      num_workers=4)
 
  return loader 

if __name__ == "__main__":
  data = video_dataloader()
  for batch in data:
    print(batch.shape)