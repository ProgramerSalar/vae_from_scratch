import torch 
from pathlib import Path

def auto_load_model(args,
                    model,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    model_ema=None,
                    optimizer_disc=None):
    
    output_dir = Path(args.output_dir)
    
    if args.auto_resume and len(args.resume) == 0:
        pass 

    if args.resume:
        pass 

    