
import torch 
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import time 
from torch.utils.data.dataloader import default_collate


from .bucket_loader import Bucketeer



class IterLoader:

    """ 
    A Wrapper to convert DataLoader as an infinite iterator.
    """

    def __init__(self,
                 dataloader: DataLoader,
                 use_distributed: bool = False, 
                 epoch: int = 0):
        
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = epoch


    @property
    def epoch(self) -> int:
        return self._epoch
    

    def __next__(self):
        try:
            data = next(self.iter_loader)

        except Exception:
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
    

def identity(x):
    return x 


def create_image_text_dataloaders(dataset,
                                  batch_size, 
                                  num_workers,
                                  multi_aspect_ratio=True,
                                  epoch=0,
                                  sizes=[(512, 512), (384, 640), (640, 384)],
                                  use_distributed=True,
                                  world_size=None,
                                  rank=None):
    

    """ 
        The dataset has already been splited by different rank
    """

    if use_distributed:
        assert world_size is not None
        assert rank is not None
        sampler = DistributedSampler(dataset=dataset,
                                     shuffle=True,
                                     num_replicas=world_size,
                                     rank=rank,
                                     seed=epoch)
        
    else:
        sampler = RandomSampler(dataset)


    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=identity if multi_aspect_ratio else default_collate,
        drop_last=True
    )

    if multi_aspect_ratio:
        dataloader_iterator = Bucketeer(dataloader=dataloader,
                                        sizes=sizes,
                                        is_infinite=True,
                                        epoch=epoch)
        
    else:
        dataloader_iterator = iter(dataloader)


    # To make it infinite 
    loader = IterLoader(dataloader=dataloader_iterator,
                        use_distributed=False,
                        epoch=epoch)
    
    return loader




if __name__ == "__main__":

    from .dataset_cls import ImageDataset
    image_dataset = ImageDataset(
                                anno_file='annotation/image_text.jsonl'
                                )
    
    dataloader = create_image_text_dataloaders(dataset=image_dataset,
                                               batch_size=1,
                                               num_workers=1,
                                               world_size=1,
                                               rank=1)
    
    print(dataloader)