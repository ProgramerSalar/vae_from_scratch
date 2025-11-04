from torch.utils.data import Dataset
import jsonlines, cv2, torch, random
from tqdm import tqdm
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, 
                anno_file, 
                resolution=256,
                max_frames=6,
                add_normalize=True):
        super().__init__()
        self.max_frames = max_frames

        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        self.video_annos = []
        for anno_file_ in anno_file:
            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    self.video_annos.append(item)

        print(f"Total Remained {len(self.video_annos)} videos")

        self.video_processor = VideoFrameProcessor(resolution=resolution,
                                                   num_frames=max_frames,
                                                   add_normalize=add_normalize)
        
    def __len__(self):
        return len(self.video_annos)
    

    def __getitem__(self, index):
        video_anno = self.video_annos[index]
        video_path = video_anno['video']
        
        try:
            video_tensors, video_frames = self.video_processor(video_path=video_path)
            assert video_tensors.shape[1] == self.max_frames
            return {
                "video": video_tensors,
                "identifier": 'video'
            }


        except:
            assert ValueError("Error...")
        

    
class VideoFrameProcessor:

    # Load a video and transform 
    def __init__(self,
                 resolution=256,
                 num_frames=24,
                 add_normalize=True,
                 sample_fps=24):
        
        image_size = resolution 
        transform_list = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size)
        ]

        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        print(f"Transform list is {transform_list}")
        self.num_frames = num_frames
        self.transform = transforms.Compose(transform_list)
        self.sample_fps = sample_fps

    def __call__(self, video_path):
        try:
            frames = []
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            print(fps)

            while True:
                flag, frame = video_capture.read()
                if not flag:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)

            video_capture.release()
            sample_fps = self.sample_fps
            interval = max(int(fps / sample_fps), 1)
            frames = frames[::interval]
            
            
            if len(frames) < self.num_frames:
                assert TypeError("Please upload the video min for 10 sec.")

            start_indexs = list(range(0, max(0, len(frames) - self.num_frames + 1)))
            start_index = random.choice(start_indexs)
            
            filtered_frames = frames[start_index : start_index + self.num_frames]
            assert len(filtered_frames) == self.num_frames, f"The sampled frames should equal to {self.num_frames}"

            filtered_frames = torch.stack(filtered_frames).float() / 255 
            filtered_frames = self.transform(filtered_frames)
            filtered_frames = filtered_frames.permute(1, 0, 2, 3)

            return filtered_frames, None






        except Exception as e:
            assert("I thing, Problem with video_path.")
        



if __name__ == "__main__":

    path = "/home/manish/Desktop/projects/vae_from_scratch/annotation/video_text.jsonl"
    video_dataset = VideoDataset(anno_file=path)
    out = video_dataset.__getitem__(0)
    print(out)