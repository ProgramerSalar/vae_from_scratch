from datasets import load_dataset, Features, Video
from torch.utils.data import DataLoader
from decord import VideoReader, cpu
from torchvision import transforms as T 
import torch 

def custom_collate(batch):

  videos = []
  for item in batch:
    path = item['video']['path']
    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)

    indices = list(range(0, min(total_frames, 64), 4))[:16]
    frames = vr.get_batch(indices).asnumpy()

    transform = T.Compose([
      T.ToPILImage(),
      T.Resize((256, 256)),
      T.ToTensor(),
      T.Normalize([0.5]*3, [0.5]*3)
    ])
    video_tensor = torch.stack([transform(f) for f in frames])
    videos.append(video_tensor)

  return {
    'video': torch.stack(videos) if len(videos) > 1 else videos[0]
  }


def video_dataloader(args):
  ds = load_dataset(
    "ProgramerSalar/video_dataset",
    split="train",
    # streaming=True,
    cache_dir="../../vae_from_scratch/train_dataset/Data")

  ds = ds.cast_column("video", Video(decode=False))
  dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=custom_collate, num_workers=0)

  return dataloader






    

if __name__ == "__main__":
  dataloader = video_dataloader()
  for batch in dataloader:
    print(batch['video'].shape)




# hf token: 