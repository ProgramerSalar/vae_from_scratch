import torch, math
import numpy as np 

import sys 
sys.path.append('../../vae_from_scratch/video_vae')
from vae.causal_vae import CausalVAE

def cosine_scheduler(total_step,
                     warmup_step,
                     max_lr=5e-5,  
                     min_lr=1e-5, 
                     start_warmup_value=1e-6,
                     ):

    
    

    decay_step = total_step - warmup_step

    warmup_schedule = np.linspace(start_warmup_value, max_lr, warmup_step+1)[:-1]

    scheduler = np.array([
            min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * i / decay_step))
        for i in range(decay_step+1)
        ])
        
    scheduler = np.concatenate((warmup_schedule, scheduler))

    return scheduler

    







if __name__ == "__main__":

    total_step = 100 
    warmup_percentage = 0.1     # Use 10% of steps for warmup 
    warmup_phase_step = int(total_step * warmup_percentage)
   

    scheduler = cosine_scheduler(total_step=total_step,
                                 warmup_step=warmup_phase_step
                                 )
    print(scheduler)
