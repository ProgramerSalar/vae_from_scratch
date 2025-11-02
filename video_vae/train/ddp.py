import torch 

from vae.causal_vae import CausalVAE
from middleware.utils import MetricLogger, SmoothedValue

def train_one_epoch(args,
                model,
                optimizer,
                optimizer_disc,
                epoch,
                lr_schedule_values,
                lr_schedule_values_disc,
                data_loader,
                loss_scaler,
                loss_scaler_disc,
                start_steps=0
            ):
    

    model.train()
    metric_logger = MetricLogger(delimiter=" ")

    if optimizer is not None:
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    if optimizer_disc is not None:
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)

    if args.model_dtype == 'bf16':
        _dtype = torch.bfloat16
    else:
      _dtype = torch.float16

    
    print(f"Start training epoch {epoch} iters per inner epoch {args.iters_per_epoch}")
    for step in metric_logger.log_every(range(args.iters_per_epoch), args.print_freq, header):

        it = start_steps + step
        
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)

        
        if optimizer_disc is not None:
            for i, param_group in  enumerate(optimizer_disc.param_groups):
                if lr_schedule_values_disc is not None:
                    param_group["lr"] = lr_schedule_values_disc[it] * param_group.get("lr_scale", 1.0)



        sample = next(iter(data_loader))
        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=_dtype):
            rec_loss, gan_loss, log_loss = model(sample, args.global_step)

        ###################################################################################
        # THe update of rec_loss 
        if rec_loss is not None:
            loss_value = rec_loss.item()

            with torch.autograd.set_detect_anomaly(True):

                optimizer.zero_grad()
                # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(rec_loss, optimizer, parameters=model.vae.parameters(), retain_graph=True)

            # This loss_logger is used when datype is float16 or float32 not for the bfloat16
            if "scaler" in loss_scaler.state_dict():
                loss_scaler_value = loss_scaler.state_dict()["scaler"]
            else:
                loss_scaler_value = 1

            metric_logger.update(vae_loss=loss_value)
            metric_logger.update(loss_scaler=loss_scaler_value)

        ############################################################################################

        # # The update of gan loss 
        if gan_loss is not None:
          gan_loss_value = gan_loss.item()

          with torch.autograd.set_detect_anomaly(True):
            optimizer_disc.zero_grad()
            disc_grad_norm = loss_scaler_disc(gan_loss, optimizer_disc, parameters=model.loss.discriminator.parameters(), retain_graph=False)
            print(f"<----------Testing line [11]----------->")
            print(f"disc_grad_norm: {disc_grad_norm}")

            # print(f"Let's see the what is the parameters has: >>>>>>>> {model.loss.discriminator}")
          

        

        





