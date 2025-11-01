import torch, math
import numpy as np 

import sys 
sys.path.append('../../vae_from_scratch/video_vae')
from vae.causal_vae import CausalVAE

# def cosine_scheduler(total_step,
#                      warmup_step,
#                      max_lr=5e-5,  
#                      min_lr=1e-5, 
#                      start_warmup_value=1e-6,
#                      ):

    
    

#     decay_step = total_step - warmup_step

#     warmup_schedule = np.linspace(start_warmup_value, max_lr, warmup_step+1)[:-1]

#     scheduler = np.array([
#             min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * i / decay_step))
#         for i in range(decay_step+1)
#         ])
        
#     scheduler = np.concatenate((warmup_schedule, scheduler))

#     return scheduler

    
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, 
        start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule







if __name__ == "__main__":

    total_step = 100 
    warmup_percentage = 0.1     # Use 10% of steps for warmup 
    warmup_phase_step = int(total_step * warmup_percentage)
   

    scheduler = cosine_scheduler(total_step=total_step,
                                 warmup_step=warmup_phase_step
                                 )
    print(scheduler)
