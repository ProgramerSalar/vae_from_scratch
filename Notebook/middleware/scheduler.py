import torch, math
import numpy as np 


def constant_scheduler(base_value,
                       epochs,
                       nither_per_ep,
                       warmup_epochs=0,
                       start_warmup_value=1e-6,
                       warmup_steps=-1):
    
    """ 
        generate a schedule of values (commonly for a learning rate) 
        that include an optional "warmup" phase followed by a constant value 
        for the remainder of the duration.

        Args:
            base_value: The constant value that the schedule will hold after the warmup phase.
            epochs: The total number of epochs for the schedule.
            nither_per_ep: The number of iteration per epoch.
            warmup_epochs: The number of epochs dedicated to the warmup phase (default 0 -> no warmup)
            start_warmup_value: the initial value at the very begining of the warmup (defaults 1e-6)
            warmup_steps: An alternative way to specify the warmup duration in terms of total steps 
                        instead of epochs. if set this overrides warmup_epochs (default -1 -> it's not used)
    """


    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * nither_per_ep

    if warmup_steps > 0:
        warmup_iters = warmup_steps

    print(f"set warmup steps = {warmup_iters}")

    if warmup_iters > 0:
        warmup_schedule = np.linspace(start=start_warmup_value,
                                      stop=base_value,
                                      num=warmup_iters)
        
    iters = epochs * nither_per_ep - warmup_iters
    schedule = np.array([base_value] * iters)
    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * nither_per_ep, ValueError
    return schedule

    






def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1):
    
    """" 
        for more info: https://arxiv.org/pdf/1608.03983
    """
    

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_steps > 0:
        warmup_iters = warmup_steps

    print(f"Set warmup steps = {warmup_iters}")

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([
        final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi  * i / (len(iters))))
        for i in iters
    ])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs  * niter_per_ep, ValueError

    return schedule



