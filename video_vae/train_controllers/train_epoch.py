import torch, math, sys
from torch.nn import Module
from typing import Iterable


from metriclogger import MetricLogger, SmoothedValue



def train_epoch(model: Module,
                    model_dtype: str,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    optimizer_disc: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    loss_scaler_disc,
                    lr_schedule_value,
                    lr_schedule_values_disc,
                    clip_grad: float = 0,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    args=None,
                    print_freq=20,
                    iters_per_epoch=2000
                    ): 
    

    """
    This function is train the model.
    
    Args:
        model (nn.Module): your vae model.
        model_dtype (str): model data type.
        data_loader (Itrable): the dataloader.
        optimizer (torch.optim.Optimizer): optimizer like (`adam`, `adamw`, etc...)
        optimizer_disc (torch.optim.Optimizer): optimizer like (`adam`, `adamw`, etc...)
        device (torch.device): model device.
        epoch (int): epoch of your model.
        loss_scaler: the loss of your model.
        loss_scaler_disc: the loss of your discriminator.
        lr_scaler_value: learning rate value (default 1e-6)
        lr_scaler_value_disc: learning rate discriminator value 
        clip_grad: the normalization of your model.
        log_writer: style you print epochs 
        lr_scheduler: scheduler funcition
        start_epoch: what is your start epoch (default 0)
        args: argument of your arch.
        print_freq (20): print frequentily to start weight save.
        iters_per_epoch (2000): how much iteration have in one epochs.
    """


    # activate the training mode.
    model.train()
    metric_logger = MetricLogger(delimiter=" ")
    

    if optimizer is not None:
        metric_logger.add_meter(name="lr", meter=SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(name="min_lr", meter=SmoothedValue(window_size=1, fmt='{value:.6f}'))

    if optimizer_disc is not None:
        metric_logger.add_meter(name="disc_lr", meter=SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(name='disc_min_lr', meter=SmoothedValue(window_size=1, fmt='{value:.6f}'))
        
    header = f"Epoch: [{epoch}]"
    _dtype = torch.bfloat16 if model_dtype == 'bf16' else torch.float16
    print(f"what is the epoch: {epoch} and iteration : {iters_per_epoch}")

    for step in metric_logger.log_every(iterable=range(iters_per_epoch), 
                                        print_freq=print_freq,
                                        header=header):
        
        if step >= iters_per_epoch:
            break

        # global iteration training.
        global_it = start_steps + step 

        if lr_schedule_value is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_schedule_value[global_it] * param_group.get("lr_scaler", 1.0)


        if optimizer_disc is not None:
            for i, param_group in enumerate(optimizer_disc.param_groups):
                param_group['lr'] = lr_schedule_values_disc[global_it] * param_group.get("lr_scaler", 1.0)


        samples = next(data_loader)
        samples['video'] = samples['video'].to(device, non_blocking=True)

        # enable automatic mix precision (amp) training 
        with torch.amp.autocast(device_type="cuda", dtype=_dtype):

            # calculate the model loss 
            print(f"i want to be know that dataset: {samples['identifier']}")
            rec_loss, gan_loss, log_loss = model(x=samples['video'],
                                                 step=args.global_step,
                                                 identifier=samples['identifier'])
            

        # the update of rec_loss 
        if rec_loss is not None:
            loss_value = rec_loss.item()

            if not math.isfinite(loss_value):
                print(f"i want to be a know that infinite value in rec_loss: {loss_value}")
                sys.exit(1)

            optimizer.zero_grad()
            # the second order deriviative provides information about the curvature of the loss surface, allowing them to take more direct steps toward the minimum.
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            grad_norm = loss_scaler(loss=rec_loss,
                                    optimizer=optimizer,
                                    clip_grad=clip_grad,
                                    parameters=model.module.vae.parameters(),
                                    create_graph=is_second_order)
            
            loss_scaler_value = loss_scaler.state_dict()["scale"] if "scale" in loss_scaler.state_dict() else 1
            metric_logger.update(vae_loss=loss_value)
            metric_logger.update(loss_scale=loss_scaler_value)
            metric_logger.update(grad_norm=grad_norm)

            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)

            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]

            metric_logger.update(weight_decay=weight_decay_value)
            



        # update the gan_loss
        if gan_loss is not None:
            gan_loss_value = gan_loss.item()

            if not math.isfinite(gan_loss_value):
                print(f"i want to be a know that infinite value in gan_loss: {gan_loss_value}")
                sys.exit(1)

            optimizer_disc.zero_grad()
            is_second_order = hasattr(optimizer_disc, 'is_second_order') and optimizer_disc.is_second_order

            disc_grad_norm = loss_scaler_disc(loss=gan_loss,
                                                optimizer=optimizer_disc,
                                                clip_grad=clip_grad,
                                                parameters=model.modules.vae.discriminator.parameters(),
                                                create_graph=is_second_order)
            
            disc_loss_scaler_value = loss_scaler_disc.state_dict()["scale"] if "scale" in loss_scaler_disc.state_dict() else 1
            metric_logger.update(disc_loss = gan_loss_value)
            metric_logger.update(disc_loss_scaler=disc_loss_scaler_value)
            metric_logger.update(disc_grad_norm=disc_grad_norm)

            min_lr = 10.
            max_lr = 0.
            for group in optimizer_disc.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(disc_lr=max_lr)
            metric_logger.update(disc_min_lr=min_lr)

        torch.cuda.synchronize()
        # update the log_loss
        new_log_loss = {
            k.split('/')[-1]:v
            if k not in ['total_loss'] else ValueError
            for k, v in log_loss.items()
        }
        metric_logger.update(**new_log_loss)

        # update the log_writer value 
        if log_writer is not None:
            print(f"work in progress...")

        if lr_scheduler is not None:
            print(f"work in progress...")

        args.global_step = args.global_step + 1 

    # gather the stats from all process.
    metric_logger.synchronize_between_processes()
    print(f"Average stats: {metric_logger}")

    return {
        k: value.global_avg
        for k, value in metric_logger.meters.items()
    }


        

        



        

            



            












            




        




    



 
    



