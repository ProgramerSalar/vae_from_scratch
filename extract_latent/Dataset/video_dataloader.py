import torch, cv2
from torch.utils.data import DataLoader
from torchvision import transforms as py_transform
from torchvision.transforms.functional import InterpolationMode


def get_transform(width, 
                  height, 
                  new_width=None, 
                  new_height=None, 
                  resize=False):
    
    transform_list = []
    if resize:
        # rescale according to the target ratio 
        scale = max(new_width / width, new_height / height)
        resized_width = round(width * scale)
        resized_heght = round(height / scale)

        transform_list.append(py_transform.Resize(size=(resized_heght, resized_heght), 
                                                  interpolation=InterpolationMode.BICUBIC, 
                                                  antialias=True

            ))
        transform_list.append(py_transform.CenterCrop((new_height, new_width)))

    transform_list.extend([
        py_transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_list = py_transform.Compose(transform_list)

    return transform_list

def load_video_and_transform(video_path,
                             frame_indexs,
                             frame_number,
                             new_width=None,
                             new_height=None,
                             resize=False):
    
    video_capture = None
    frame_indexs_set = set(frame_indexs)
    
    try:
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        frame_index=0
        while True:
            flag, frame = video_capture.read()
            if not flag:
                break
            if frame_index > frame_indexs[-1]:
                break
            if frame_index in frame_indexs_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)  #[channels, height, width]
                frames.append(frame)
            frame_index += 1 
        video_capture.release()

        if len(frames) == 0:
            print(f"empty video: {video_path}")
            return None
        
        frames = frames[:frame_number]
        duration = ((len(frames) - 1) // 8) * 8 + 1  # make sure the frames match f*8+1
        frames = frames[:duration]
        frames = torch.stack(frames).float() / 255
        
        video_transform = get_transform(width=frames.shape[-1],
                                        height=frames.shape[-2],
                                        new_height=new_height,
                                        new_width=new_width,
                                        resize=resize)
        
        frames = video_transform(frames).permute(1, 0, 2, 3)
        return frames

    except ValueError:
        pass





if __name__ == "__main__":

    # tranform = get_transform(width=512, height=512,
    #                          new_height=256, new_width=256,
    #                          resize=True)
    # print(tranform)
    # -------------------------
    num_frames= 121

    load_video_and_transform_ = load_video_and_transform(video_path="../../vae_from_scratch/Data/webvid101/stock-footage-western-clownfish-swimming-around.mp4",
                                                         frame_indexs=list(range(num_frames)),
                                                         frame_number=num_frames,
                                                         new_width=256,
                                                         new_height=256,
                                                         resize=True
                                                         )
    print(load_video_and_transform_.shape)

