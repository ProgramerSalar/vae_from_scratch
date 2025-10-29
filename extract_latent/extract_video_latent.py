import torch 
from torch.utils.data import DataLoader
from concurrent import futures
import sys 

from args import get_args

sys.path.append('../../vae_from_scratch')
from video_vae.vae.wrapper import CausalVideoLossWrapper
sys.path.append('../../vae_from_scratch')
from Dataset.video_dataset import VideoDataset


def build_model(args):
    print(f"Let's know the args is printed or not: >>>>> {args}")
    
    model = CausalVideoLossWrapper(num_groups=args.batch_size)
    return model



def build_data_loader(args):

    def collate_fn(batch):

        return_batch = {
            'input': [],
            'output': []
        }

        for videos_ in batch:
            for video_input in videos_:
                return_batch['input'].append(video_input['input'])
                return_batch['output'].append(video_input['output'])

        return return_batch
    

    dataset = VideoDataset(anno_file=args.anno_file,
                           width=args.width,
                           height=args.height,
                           num_frames=args.num_frames)
    
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        num_workers=6,
                        pin_memory=True,
                        shuffle=False,
                        collate_fn=collate_fn,
                        drop_last=False,
                        )
    
    return loader




def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)

    if args.model_dtype == "bf16":
        torch_dtype = torch.bfloat16

    else:
        torch_dtype = torch.float16

    data_loader = build_data_loader(args)
    task_queue = []

    with futures.ThreadPoolExecutor(max_workers=16) as Excecuter:

        for sample in data_loader:
            input_video_list = sample['input']
            output_path_list = sample['output']

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch_dtype):
                for video_input, output_path in zip(input_video_list, output_path_list):
                    video_latent = model.encode_latent(video_input.to(device), 
                                                       sample=True,  # <<<---- this function is not defined...
                                                       )
                    task_queue.append(Excecuter.submit(save_tensor, video_latent, output_path))

        for future in futures.as_completed(task_queue):
            res = future.result()


    


def save_tensor(tensor, output_path):
    try:
        torch.save(tensor.clone(), output_path)

    except ValueError:
        print("you Tensor is not saved.")



if __name__ == "__main__":
    args = get_args()
    main(args=args)