import torch 
from torch.utils.data import DataLoader

from .video_dataset import VideoDataset

def Video_dataloader(args):

    dataset = VideoDataset(video_dir='../../vae_from_scratch/Data/train_dataset')
    train_video_dataloader = DataLoader(dataset=dataset,
                                        batch_size=args.batch_size)
    
    return train_video_dataloader

