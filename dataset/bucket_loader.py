import torch, random
from torch.utils.data import DataLoader




class Bucketeer:

    def __init__(self,
                 dataloader,
                 sizes = [(256, 256),
                          (192, 384),
                          (192, 320),
                          (384, 192),
                          (320, 192)],
                is_infinite=True,
                epoch=0
                ):
        
        super().__init__()
        self.sizes = sizes
        # {(256, 256): [], (192, 384): [], (192, 320): [], (384, 192): [], (320, 192): []}
        self.buckets = {s: [] for s in self.sizes}
        self.batch_size = dataloader.batch_size
        self.iterator = iter(dataloader)
        # print(self.iterator)    # <torch.utils.data.dataloader._SingleProcessDataLoaderIter object at 0x715f92a60be0>
        print(next(self.iterator))

        


    def get_available_batch(self):

        available_size = []
        for b in self.buckets:
            if len(self.buckets[b]) >= self.batch_size:
                available_size.append(b)
        
        if len(available_size) == 0:
            return None
        
        else:
            print("work in progress...")

            


    def __next__(self):
        batch = self.get_available_batch()
        print(batch)


    





    




if __name__ == "__main__":
    from dataset_cls import ImageDataset

    image_dataset = ImageDataset(anno_file="/home/manish/Desktop/projects/vae_from_scratch/annotation/image_text.jsonl")


    data_loader = DataLoader(dataset=image_dataset,
                             batch_size=2,
                             shuffle=True)
    
    out = Bucketeer(dataloader=data_loader)

    # print(out.get_available_batch())
    print(out.__next__())