import torch 
from typing import Iterable
from utils import MetricLogger, SmoothedValue
import math, sys




def train_one_epoch(
        model: torch.nn.Module,
        model_dtype: str,
        data_laoder: Iterable,
        optimizer: torch.optim.Optimizer,
        optimizer_disc: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        loss_scaler_disc,
        clip_grad: float = 0,
        log_writer=None,
        lr_scheduler=None,
        start_steps=None,
        lr_schedule_values=None,
        lr_schedule_values_disc=None,
        args=None,
        print_freq=20,
        iters_per_epoch=2000
    ):


    # The trainer for causal video vae 
    model.train()
    metric_logger = MetricLogger(delimiter=" ")

    if optimizer is not None:
        metric_logger.add_meter(name='lr', 
                                meter=SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(name='min_lr',
                                meter=SmoothedValue(window_size=1, fmt='{value:.6f}'))

    if optimizer_disc is not None:
        metric_logger.add_meter(name='disc_lr',
                                meter=SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(name='disc_min_lr',
                                meter=SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f"Epoch: [{epoch}]"

    _dtype = torch.bfloat16 if model_dtype == 'bf16' else torch.float16
    print(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")

    for step in metric_logger.log_every(range(iters_per_epoch), 
                                        print_freq=print_freq, 
                                        header=header):
        
        if step >= iters_per_epoch:
            break

        it = start_steps + step # global training iteration 
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get('lr_scale', 1.0)


        if optimizer_disc is not None:
            for i, param_group in enumerate(optimizer_disc.param_groups):
                if lr_schedule_values_disc is not None:
                    param_group['lr'] = lr_schedule_values_disc[it] * param_group.get("lr_scale", 1.0)

        samples = next(data_laoder)
        samples['video'] = samples['video'].to(device, non_blocking=True)

        with torch.amp.autocast(enabled=True, dtype=_dtype):
            rec_loss, gan_loss, log_loss = model(samples['video'], args.global_step, identifier=samples['identifier'])

        ############################################################################################################

        # the update of rec_loss 
        if rec_loss is not None:
            loss_value = rec_loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.", force=True)
                sys.exit(1)

            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(rec_loss,
                                    optimizer,
                                    clip_grad=clip_grad,
                                    parameters=model.module.vae.parameters(),
                                    create_graph=is_second_order)
            

            if "scale" in loss_scaler.state_dict():
                loss_scaler_value = loss_scaler.state_dict()["scale"]

            else:
                loss_scaler_value = 1 


            metric_logger.update(vae_loss=loss_value)
            metric_logger.update(loss_scaler=loss_scaler_value)




        ############################################################################################################

        # the update of gan_loss 
        if gan_loss is not None:
            
            gan_loss_value = gan_loss.item()

            if not math.isfinite(gan_loss_value):
                print(f"The gan discriminator Loss is {gan_loss_value}, stoppig training", force=True)
                sys.exit(1)

            optimizer_disc.zero_grad()
            is_second_order = hasattr(optimizer_disc, 'is_second_order') and optimizer_disc.is_second_order 
            disc_grad_norm = loss_scaler_disc(gan_loss,
                                              optimizer_disc,
                                              clip_grad=clip_grad,
                                              parameters=model.module.loss.discriminator.parameters(),
                                              create_graph=is_second_order)
            
            if "scale" in loss_scaler_disc.state_dict():
                disc_loss_scale_value = loss_scaler_disc.state_dict()["scale"]
            else:
                disc_loss_scale_value = 1 

            metric_logger.update(disc_loss=gan_loss_value)
            metric_logger.update(disc_loss_scale=disc_loss_scale_value)
            metric_logger.update(disc_grad_norm=disc_grad_norm)

            min_lr = 10.
            max_lr = 0.
            for group in optimizer_disc.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(disc_lr=max_lr)
            metric_logger.update(disc_min_lr=min_lr)

        torch.cuda.synchronize()
        new_log_loss = {
            k.split('/')[-1]: v
            for k, v in log_loss.items()
            if k not in ["total_loss"]
        }
        metric_logger.update(**new_log_loss)

        if rec_loss is not None:

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
            metric_logger.update(grad_norm=grad_norm)

        


        if log_writer is not None:
            log_writer.update(**new_log_loss, head="train/loss")
            log_writer.update(lr=max_lr, 
                              head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_setup()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)


        args.global_step = args.global_step + 1 

    # gather the stats from all process 
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")


    return {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
    }


        
   



    
