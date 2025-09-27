import torch 
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.data.dataloader import default_collate
import time 

# from bucket_loader import Bucketeer


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
        self._epoch = epoch
        self._use_distributed = use_distributed

    def __next__(self):
        try:
            
            data = next(self.iter_loader)

        

        except Exception as e:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

            
            

        return data 
    

    def __iter__(self):
        return self 
    
    def __len__(self):
        return len(self._dataloader)
    





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

    assert world_size is not None, "make sure `world_size` is not None"
    assert rank is not None, "make sure `rank is not None"

    image_gpus = max(1, int(world_size * image_mix_ratio))
    if use_image_video_mixed_training:
        video_gpus = world_size - image_gpus
    else:
        # only use video data 
        video_gpus = world_size
        image_gpus = 0 

    if rank < video_gpus:
        sampler = DistributedSampler(
            dataset=dataset,
            shuffle=True,
            num_replicas=image_gpus,
            rank=rank - video_gpus,
            seed=epoch
        )
    else:
        sampler = DistributedSampler(
            dataset=dataset,
            shuffle=True,
            num_replicas=image_gpus,
            rank=rank - video_gpus,
            seed=epoch
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=default_collate,
        drop_last=True
    )

    # TO make it infinite 
    loader = IterLoader(dataloader=loader,
                        use_distributed=True,
                        epoch=epoch)
    

    return loader






    

    





if __name__ == "__main__":
    from dataset_cls import ImageDataset
    

    image_dataset = ImageDataset(anno_file="/home/manish/Desktop/projects/vae_from_scratch/annotation/image_text.jsonl")
    # data_laoder = DataLoader(dataset=image_dataset,
    #                          batch_size=2)
    
    # out = IterLoader(dataloader=data_laoder)
    # print(f"out: {out.__next__()}")

    # -----------------------------------------------------------
    out = create_mixed_dataloaders(dataset=image_dataset,
                                   batch_size=2,
                                   num_workers=2,
                                   world_size=1,
                                   rank=0)
    print(out)
    


    # out = create_image_text_dataloader(dataset=image_dataset,
    #                                    batch_size=2,
    #                                    num_workers=2,
    #                                    use_distributed=False,
    #                                    multi_aspect_ratio=False)
    # print(out)