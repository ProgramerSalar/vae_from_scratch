import torch 
from torch.utils.data import DataLoader

from .video_dataset import VideoDataset

def Video_dataloader(args):

    # Please you should download the dataset huggingface `path` and download the dataset on `train_dataset` dir
    # path: https://huggingface.co/datasets/ProgramerSalar/video_dataset/tree/main
    video_dir = "../../vae_from_scratch/Data/train_dataset/video_dataset/OpenVid_part108"
    dataset = VideoDataset(video_dir=video_dir)
    train_video_dataloader = DataLoader(dataset=dataset,
                                        batch_size=args.batch_size)
    
    return train_video_dataloader

