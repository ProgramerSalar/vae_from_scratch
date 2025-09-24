import torch 
from torch.utils.data import DataLoader






class Bucketeer:

    def __init__(self,
                 dataloader,
                 sizes=[(256, 256), (192, 384), (384, 320), (384, 192), (320, 192)],
                 is_infinte=True,
                 epoch=0
                 ):
        
        super().__init__()
        self.sizes = sizes
        self.batch_size = dataloader.batch_size
        self.buckets = {s: [] for s in self.sizes}


    def get_available_batch(self):
        
        available_size = []
        # print(self.buckets)
        for b in self.buckets:
            # print(b)
            print(self.buckets[b])
            
       
        if len(available_size) == 0:
            return None
        else:
            print("working...")



    def __next__(self):
        batch = self.get_available_batch()
        print(batch)



if __name__ == "__main__":

    from dataset_cls import ImageDataset
    image_dataset = ImageDataset(anno_file="/home/manish/Desktop/projects/vae_from_scratch/annotation/image_text.jsonl")
    # print(image_dataset)
    data_loader = DataLoader(dataset=image_dataset,
                             batch_size=2,
                             shuffle=True)
    # print(data_loader)
    out = Bucketeer(dataloader=data_loader)
    print(out.__next__())