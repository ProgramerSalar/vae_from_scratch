import torch, math
import numpy as np 



def cosine_scheduler(base_value: float = 5e-5,
                     final_value: float = 1e-5,
                     epochs: int = 100,
                     num_iter_per_epoch: int = 2000,
                     warmup_epochs: int = 5,
                     start_warmup_value:float = 1e-6,
                     warmup_steps=-1):
    
    """
        A leraning rate (LR) scheduler dynamically adjusts the learning rate during training,
        and the `cosine_scheduler` is highly effective, modern approach to this.
        for more info: https://arxiv.org/pdf/1608.03983

        Args:
            base_value: the initial value of the schedule after the warmup-up (e.g., initial learning rate.)
            final_value: the minimum value of the schedule to decay (e.g., minimum learning rate)
            epochs: total number of training epochs.
            num_iter_per_epoch: the number of iteration per epoch
            warmup_epoch: the number of epoch to dedicated to the warmup phase (default 0, no warmup)
            start_warmup_value: the starting value for the warmup phase (default 0)
            warmup_steps: An explicit number of warmup step/iterator (default -1, not warmup_steps)

        
        **Q. so question is  my total epoch 10 then why warmup epochs is 2 ?**
        
        When you set **Total Epochs = 10** and **Warmup Epochs = 2**, you are essentially dividing your training into two phases:
        1.  **Warm-up Phase (Epochs 1 and 2):** The learning rate **linearly increases** from your starting value (1e-7) up to the maximum value, or **base value** (1e-4). This takes the first **200 iterations** (2 * 100). This phase stabilizes initial training.
        2.  **Cosine Decay Phase (Epochs 3 through 10):** The learning rate **smoothly decays** from the maximum value (1e-4) down to the minimum value (1e-7). This takes the remaining **800 iterations** (1000 total - 200 warm-up).

        The reason you choose a smaller number for warm-up (e.g., 2 epochs) relative to the total (10 epochs) is that the warm-up phase is only necessary to **stabilize the very beginning of training**. It should be long enough to get the model's optimizers (like Adam) working correctly but short enough so that the model can spend the vast majority of its time in the effective **cosine decay phase** where the actual learning and fine-tuning happens.
    """

    # store the value for the warm-up phase.
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * num_iter_per_epoch       # 5*2000 => 10000

    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print(f"how much warmup iteration: {warmup_iters}")

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value,   # 1e-6
                                      base_value,           # 5e-5
                                      warmup_iters)         # 10000
        
    iters = np.arange(epochs * num_iter_per_epoch - warmup_iters)   # 100 * 2000 - 10000 => 190000
    
    scheduler = np.array([
        final_value + 0.5 \
        * (base_value - final_value) \
        * (1 + math.cos(math.pi * i / (len(iters))))
        for i in iters
    ])

    scheduler = np.concatenate(arrays=(warmup_schedule, scheduler))

    # 190000 == 200000
    assert len(scheduler) == epochs * num_iter_per_epoch, ValueError
    
    return scheduler






    

    


    
