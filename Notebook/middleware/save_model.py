import torch 
from pathlib import Path
from video_vae.utils import get_rank


def is_main_process():
    return get_rank() == 0 


def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)





def save_model(args,
               epoch,
               model,
               model_without_ddp,
               optimizer,
               loss_scaler,
               model_ema=None,
               optimizer_disc=None,
               save_ckpt_freq=1):
    

    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    checkpoint_paths = [output_dir/'checkpoint.pth']
    if epoch == 'best':
        checkpoint_paths = [output_dir / (f'checkpoint-{epoch_name}.pth')]

    if (epoch + 1) % save_ckpt_freq == 0:
        checkpoint_paths.append(output_dir / (f'checkpoint-{epoch_name}.pth'))

    
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'epoch': epoch,
            'step': args.global_step,
            'args': args
        }

        if optimizer is not None:
            to_save['optimizer'] = optimizer.state_dict()

        if loss_scaler is not None:
            to_save['scaler'] = loss_scaler.state_dict()

        if model_ema is not None:
            to_save['model_ema'] = model_ema.state_dict()

        if optimizer_disc is not None:
            to_save['optimizer_disc'] = optimizer_disc.state_dict()

        save_on_master(to_save, checkpoint_path)



