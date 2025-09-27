import torch, math, jsonlines, numpy as np, cv2, random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import resize, center_crop
from PIL import Image 
from tqdm import tqdm


class ImageTextDataset(Dataset):

    def __init__(self,
                 anno_file,
                 add_normalize=True,
                 ratios=[1/1, 3/5, 5/3],
                 sizes=[(1024, 1024), (768, 1280), (1280, 768)],
                 crop_mode='random',
                 p_random_ratio=0.0
                 ):
        
        super().__init__()
        assert crop_mode in ['random', 'center'], "make sure choose ['random', or 'center'] crop mode."
        self.p_random_ratio = p_random_ratio
        self.ratio = ratios
        self.sizes = sizes
        self.crop_mode = crop_mode


        self.image_annos = []
        if not isinstance(anno_file, list):
            anno_file = [anno_file]
            
        for anno_file_ in anno_file:
            with jsonlines.open(anno_file_, 'r') as reader:
                for item in reader:
                    self.image_annos.append(item)

        transform_list = [
            transforms.ToTensor()
        ]
        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                       std=(0.5, 0.5, 0.5)))
            
        self.transform = transforms.Compose(transform_list)


    def get_closest_size(self, x):
        
        if self.p_random_ratio > 0 and \
            np.random.randint(len(self.ratio)) < self.p_random_ratio:
            
            print('working in progress...')

        else:
            w, h = x.width, x.height 
            best_size_idx = np.argmin(a=[abs(w/h-r) for r in self.ratio])
            
        return self.sizes[best_size_idx]
    

    def get_resize_size(self,
                        original_size,
                        target_size):
        
        
        
        # 768/1280-1 = -0.4, 300/300-1 = 0
        if (target_size[1]/target_size[0] -1) * (original_size[1] / original_size[0] -1) >= 0:
            
            alt_min = int(math.ceil(max(target_size) * min(original_size) / max(original_size)))
            resize_size = max(alt_min, min(target_size))
            

        else:
            print("work in progress....")

        return resize_size



    def __len__(self):
        return len(self.image_annos)
    

    def __getitem__(self, 
                    index):
        
        image_anno = self.image_annos[index]
        

        try:
            
            img = Image.open(image_anno['image']).convert("RGB")
            text = image_anno['text']
            assert isinstance(text, str), "make sure text is string in annotation file."

            size = self.get_closest_size(img)
            resize_size = self.get_resize_size(original_size=(img.width, img.width),
                                               target_size=size)
            img = resize(img=img,
                         size=resize_size,
                         interpolation=transforms.InterpolationMode.BICUBIC,
                         antialias=True)
            
            if self.crop_mode == "center":
                img = center_crop(img=img,
                                  output_size=(size[1], size[0]))
                
            if self.crop_mode == "random":
                img = transforms.RandomCrop(size=(size[1], size[0]))(img)

            image_tensor = self.transform(img)
            
            return {
                "video": image_tensor,
                "text": text,
                "identifier": "image"
            }
        

        except Exception as e:
            print("Getting a error : {e}")




class LengthGroupedVideoTextDataset(Dataset):
    
    def __init__(self,
                 anno_file,
                 max_frames=16,
                 resolution='384p',
                 load_vae_latent=True,
                 load_text_fea=True):
        super().__init__()
        assert load_vae_latent, "Now only support loading vae latents, we will support to directly load video frames in the future."
        self.resolution = resolution
        self.max_frames = max_frames
        self.load_text_fea = load_text_fea




        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        self.video_annos = []
        for anno_file_ in anno_file:
            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    self.video_annos.append(item)

    def __len__(self):
        return len(self.video_annos)
    

    def __getitem__(self,
                    index):
        
        try:
            video_anno = self.video_annos[index]
            text = video_anno['text']
            latent_path = video_anno['latent']
            latent = torch.load(latent_path, map_location='cpu')
            print(latent.shape)
            
            if self.resolution == '384p':
                assert latent.shape[-1] == 640 // 8
                assert latent.shape[-2] == 384 // 8

            if self.resolution == '768p':
                assert latent.shape[-1] == 1280 // 8
                assert latent.shape[-2] == 768 // 8

            cur_temp = latent.shape[2]
            cur_temp = min(cur_temp, self.max_frames)
            
            video_latent = latent[:, :, :cur_temp].float()
            assert video_latent.shape[1] == 16, "make sure video frame are 16 to choosed"

            if self.load_text_fea:
                
                text_feature_path = video_anno['text_fea']
                text_feature = torch.load(text_feature_path, map_location='cpu')

                return {
                    'video': video_latent,
                    'prompt_embed': text_feature['prompt_embed'],
                    'prompt_attention_mask': text_feature['prompt_attention_mask'],
                    'pooled_prompt_embed': text_feature['pooled_prompt_embed'],
                    'identifier': 'video'
                }
            else:
                
                print('work in progress...')

        
        except Exception as e:
            print("Getting an error: {e}")



class VideoFrameProcessor:

    # load a video and transform 
    def __init__(self,
                 resolution=256,
                 num_frames=24,
                 add_normalize=True,
                 sample_fps=24):
        
        super().__init__()
        self.sample_fps = sample_fps
        self.num_frames = num_frames



        transform_list = [
            transforms.Resize(size=resolution,
                              interpolation=transforms.InterpolationMode.BICUBIC, 
                              antialias=True),
            transforms.CenterCrop(resolution)
        ]

        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                       std=(0.5, 0.5, 0.5)))
            
        self.transform = transforms.Compose(transform_list)
        


    def __call__(self,
                 video_path):
        
        try:
            
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            
            frames = []
            while True:
                flag, frame = video_capture.read()
                if not flag:
                    break
                
                frame = cv2.cvtColor(src=frame,
                                     code=cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)

            video_capture.release()
            sample_fps = self.sample_fps
            interval = max(int(fps / sample_fps), 1)
            frames = frames[::interval]
            
            
            if len(frames) < self.num_frames:
                num_frame_to_pack = self.num_frames - len(frames)
                recurrent_num = num_frame_to_pack // len(frames)
                frames = frames + recurrent_num * frames + frames[:(num_frame_to_pack % len(frames))]
                assert len(frames) >= self.num_frames, f"{len(frames)}"

            
            start_indexs = list(range(0, max(0, len(frames) - self.num_frames + 1)))
            start_index = random.choice(start_indexs)

            filtered_frames = frames[start_index : start_index + self.num_frames]
            assert len(filtered_frames) == self.num_frames, f"The sampled frames should equals to {self.num_frames}"

            filtered_frames = torch.stack(filtered_frames).float() / 255 
            filtered_frames = self.transform(filtered_frames)
            filtered_frames = filtered_frames.permute(1, 0, 2, 3)
            
            return filtered_frames, None


        except Exception as e:
            print("Getting Error: {e}")



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

        self.video_processor = VideoFrameProcessor(resolution=resolution,
                                                   num_frames=max_frames,
                                                   add_normalize=add_normalize)
        
    def __len__(self):
        return len(self.video_annos)
    

    def __getitem__(self,
                    index):
        
        video_anno = self.video_annos[index]
        video_path = video_anno['video']

        try:
            video_tensors, video_frames = self.video_processor(video_path)
            assert video_tensors.shape[1] == self.max_frames

            return {
                "video": video_tensors,
                "identifier": "video"
            }

        except Exception as e:
            print("Getting an error: {e}")



class ImageDataset(Dataset):

    def __init__(self,
                 anno_file,
                 resolution=256,
                 max_frames=8,
                 add_normalize=True):
        super().__init__()
        self.max_frames = max_frames


        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        image_paths = []
        for anno_file_ in anno_file:
            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    image_paths.append(item['image'])
        
        self.image_annos = []
        for idx in range(0, len(image_paths), self.max_frames):
            image_path_shard = image_paths[idx: idx + self.max_frames]
            
            if len(image_path_shard) < self.max_frames:
                image_path_shard = image_path_shard + image_paths[:self.max_frames - len(image_path_shard)]

            assert len(image_path_shard) == self.max_frames
            self.image_annos.append(image_path_shard)

        
        transform_list = [
            transforms.Resize(size=resolution,
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ]

        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                       std=(0.5, 0.5, 0.5)))
            
        self.transform = transforms.Compose(transform_list)



    def __len__(self):
        return len(self.image_annos)
    

    def __getitem__(self,
                    index):
        
        image_paths = self.image_annos[index]

        try:
            packed_pil_frames = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            filtered_frames = [self.transform(frame) for frame in packed_pil_frames]
            filtered_frames = torch.stack(filtered_frames)
            # print(filtered_frames.shape)    # [t, c, h, w]
            filtered_frames = filtered_frames.permute(1, 0, 2, 3)

            return {
                "video": filtered_frames,
                "identifier": 'image'
            }
            

        except Exception as e:
            print("Getting an error: {e}")


        
        



if __name__ == "__main__":
    # out = ImageTextDataset(anno_file="annotation/image_text.jsonl")
    # print(out.__getitem__(index=1))
    # -----------------------------------------------------------------------------------------
    # out = LengthGroupedVideoTextDataset(anno_file="/home/manish/Desktop/projects/vae_from_scratch/annotation/video_text.jsonl")
    # print(out.__getitem__(0))
    # ---------------------------------------------------------------
    # out = VideoFrameProcessor()
    # print(out.__call__(video_path="webvid10m/train/010451_010500/23388121.mp4"))
    # -----------------------------------
    # out = VideoDataset(anno_file="/home/manish/Desktop/projects/vae_from_scratch/annotation/video_text.jsonl")
    # print(out.__getitem__(0))
    # ------------------------------------------------------------
    # out = ImageDataset(anno_file="annotation/image_text.jsonl")
    # print(out.__getitem__(0))
    # ------------------------------------------------------------------------------
    pass 