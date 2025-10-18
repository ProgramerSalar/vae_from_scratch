import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import os
import glob
from PIL import Image

class VideoDataset(Dataset):
    """
    A PyTorch Dataset for loading video files and sampling frames.
    """
    def __init__(self, video_dir, num_frames=16, transform=None):
        """
        Args:
            video_dir (str): Path to the directory containing video files.
            num_frames (int): The number of frames to sample from each video.
            transform (callable, optional): Optional transform to be applied on a frame.
        """
        self.video_dir = video_dir
        # Find all video files in the directory with common extensions
        self.video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
        self.video_files.extend(glob.glob(os.path.join(video_dir, '*.avi')))
        self.video_files.extend(glob.glob(os.path.join(video_dir, '*.mov')))

        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        """Returns the total number of videos in the dataset."""
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Fetches the video at the given index, samples frames, and returns them.

        Args:
            idx (int): The index of the video to fetch.

        Returns:
            torch.Tensor: A tensor of shape (T, C, H, W) where T is num_frames.
        """
        video_path = self.video_files[idx]
        
        # Use our helper function to extract frames
        frames = self._extract_frames(video_path)
        
        # Apply transformations to each frame
        if self.transform:
            # We need to apply the transform to each frame in the list
            transformed_frames = [self.transform(frame) for frame in frames]
            # Stack the frames into a single tensor
            video_tensor = torch.stack(transformed_frames)
        else:
            # If no transform, we still need to convert PIL images to a tensor
            # You might want to define a default transform for this case
            to_tensor = transforms.ToTensor()
            tensor_frames = [to_tensor(frame) for frame in frames]
            video_tensor = torch.stack(tensor_frames)

        # The output shape will be (num_frames, channels, height, width)
        # e.g., torch.Size([16, 3, 224, 224])
        return video_tensor

    def _extract_frames(self, video_path):
        """
        Extracts `num_frames` evenly spaced frames from a video file.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            # Create indices for evenly spaced frames
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i.item())
                ret, frame = cap.read()
                if ret:
                    # OpenCV reads frames in BGR format, convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert NumPy array to PIL Image for torchvision transforms
                    frame_pil = Image.fromarray(frame_rgb)
                    frames.append(frame_pil)
                else:
                    # If a frame can't be read, you might want to duplicate the last one
                    if frames:
                        frames.append(frames[-1])

        cap.release()
        
        # If the video was shorter than num_frames, duplicate the last frame
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                # Handle empty or unreadable video by creating a blank frame
                blank_frame = Image.new('RGB', (224, 224), 'black')
                frames.append(blank_frame)

        return frames

# --- HOW TO USE IT ---

if __name__ == '__main__':
    # 1. Define transformations for the frames
    #    Commonly includes resizing, cropping, and normalizing.
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

   

    # 3. Instantiate the Dataset
    video_dataset = VideoDataset(video_dir='/home/manish/Desktop/projects/vae_from_scratch/vae_from_scratch/Data', num_frames=16, transform=data_transform)
    print(f"Dataset created with {len(video_dataset)} videos.")


    # 4. Create the DataLoader
    #    This handles batching, shuffling, and parallel data loading.
    data_loader = DataLoader(video_dataset, batch_size=2, shuffle=True, num_workers=2)


    # 5. Iterate through the data
    print("\nIterating through one batch of data:")
    for batch in data_loader:
        # The 'batch' is a tensor containing the video frames
        # Shape: (batch_size, num_frames, channels, height, width)
        print(f"Batch shape: {batch.shape}")
        # Example: torch.Size([2, 16, 3, 224, 224])
        
        # You can now feed this batch to your model
        # e.g., output = your_video_model(batch)
        
        # We'll just break after the first batch for this example
        break