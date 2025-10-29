import torch 
from torch import nn 
from torch.utils.data import Dataset
import jsonlines

from .video_dataloader import load_video_and_transform

class VideoDataset(Dataset):

    def __init__(self,
                 anno_file,
                 width,
                 height,
                 num_frames):
        
        super().__init__()
        self.annotation = []
        self.width = width
        self.height = height
        self.num_frames = num_frames

        with jsonlines.open(anno_file, 'r') as reader:
            for item in reader:
                self.annotation.append(item)

        total_len = len(self.annotation)
        print(f"Total {len(self.annotation)} Videos.")

    def __len__(self):
        return len(self.annotation)
    

    def __getitem__(self, index):
        try:
            video_item = self.annotation[index]
            videos_per_task = self.process_one_video(video_item)
            
        except:
            pass

        return videos_per_task


    def process_one_video(self, video_item):
        
        video_path = video_item['video']
        output_latent_path = video_item['latent']
        videos_per_task = []
        
        frame_indexs = list(range(self.num_frames))
        try:
            
            video_frames_tensors = load_video_and_transform(video_path=video_path,
                                                            frame_indexs=frame_indexs,
                                                            frame_number=self.num_frames,
                                                            new_width=self.width,
                                                            new_height=self.height,
                                                            resize=True)
            if video_frames_tensors is None:
                return videos_per_task
            

            video_frames_tensors = video_frames_tensors.unsqueeze(0)
            videos_per_task.append({
                'video': video_path,
                'input': video_frames_tensors,
                'output': output_latent_path
            })

        except ValueError:
            pass 


        return videos_per_task

    


        






if __name__ == "__main__":
    
    video_dataset = VideoDataset(anno_file="../../vae_from_scratch/annotation/video_text.jsonl",
                                 width=256,
                                 height=256,
                                 num_frames=121)
    print(video_dataset.__getitem__(0))
