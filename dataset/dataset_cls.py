import torch 
from torch.utils.data import Dataset, DataLoader
import glob, os, cv2
from PIL import Image
from torchvision import transforms
from einops import rearrange

class VideoDataset(Dataset):

    def __init__(self,
                 video_dir,
                 num_frames=16,
                 ):
        
        self.video_files = glob.glob(pathname=(os.path.join(video_dir, '*.mp4')))
        self.num_frames = num_frames
        

        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_files)
    

    def _extract_frames(self, video_path):

        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:

            # [0, 1573-1]
            indices = torch.linspace(0, total_frames-1, self.num_frames).long()
            
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i.item())
                ret, frame = cap.read()
                if ret:
                    # OpenCv reads frames in BGR format, convert to RGB 
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # convert numpy array to PIL Image for torchvision transform 
                    frame_pil = Image.fromarray(frame_rgb)
                    frames.append(frame_pil)

        cap.release()

        # If video was shorter then num_frames, duplicate the last frame 
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])

            
        return frames
    
    def __getitem__(self, idx):
        
        video_path = self.video_files[idx]
        frames = self._extract_frames(video_path)

        if self.transform:
            transformed_frames = [self.transform(frame) for frame in frames]
            video_tensor = torch.stack(transformed_frames)
            video_tensor = rearrange(video_tensor, 't c h w -> c t h w')
            video_tensor = video_tensor.contiguous()

        return video_tensor
        


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoDataset(video_dir="../../Data/train_dataset")
    # dataset.__getitem__(0)

    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=1,
                                  num_workers=4)
    
    for data in train_dataloader:
        print(data.shape)