import torch 
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.data.dataloader import default_collate

from bucket_loader import Bucketeer


class IterLoader:

    """ 
    A wrapper to convert DataLoader as an infinite iterator
    """

    def __init__(self,
                 dataloader: DataLoader,
                 use_distributed: bool = False, 
                 epoch: int = 0):
        
        super().__init__()
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            
            data = next(self.iter_loader)

        except Exception as e:
            print("Exception...")





def create_image_text_dataloader(dataset,
                                 batch_size,
                                 num_workers,
                                 multi_aspect_ratio=True,
                                 epoch=0,
                                 sizes=[(512, 512), (384, 640), (640, 384)],
                                 use_distributed=True,
                                 world_size=None,
                                 rank=None
                                ):
    
    if use_distributed:
        print('work in progress...')
        
    else:
        sampler = RandomSampler(data_source=dataset)


    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=identity if multi_aspect_ratio else default_collate,
        pin_memory=True,
        drop_last=True
    )

    if multi_aspect_ratio:
        # dataloader_iterator = Bucketeer(dataloader=dataloader,
        #                                 epoch=epoch)
        # print(dataloader_iterator)
        print("work in progress...")

    else:
        dataloader_iterator = iter(dataloader)

    loader = IterLoader(dataloader=dataloader_iterator,
                        use_distributed=False,
                        epoch=epoch)
    
    return loader



def identity(x):
    return x 


def create_length_grouped_video_text_dataloader(dataset,
                                                batch_size,
                                                num_workers,
                                                max_frames,
                                                world_size=None,
                                                rank=None,
                                                epoch=0,
                                                use_distributed=False):
    
    if use_distributed:
        print('work in progress...')

    else:
        sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
        drop_last=True
    )

    return dataloader



def create_mixed_dataloaders(
        dataset,
        batch_size,
        num_workers,
        world_size=None,
        rank=None,
        epoch=0,
        image_mix_ratio=0.1,
        use_image_video_mixed_training=True
    ):
    

    """Image and video mixed training dataloader"""

    print('work in progress...')

    

    





if __name__ == "__main__":
    from dataset_cls import ImageDataset
    

    image_dataset = ImageDataset(anno_file="/home/manish/Desktop/projects/vae_from_scratch/annotation/image_text.jsonl")


    out = create_image_text_dataloader(dataset=image_dataset,
                                       batch_size=2,
                                       num_workers=2,
                                       use_distributed=False,
                                       multi_aspect_ratio=False)
    print(out)