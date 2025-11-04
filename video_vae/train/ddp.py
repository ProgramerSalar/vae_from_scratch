import torch, math, sys

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
                start_steps=0,
                log_writer=None
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

    print_freq = args.print_freq

    
    print(f"Start training epoch {epoch} iters per inner epoch {args.iters_per_epoch}")
    for step in metric_logger.log_every(
        iterable=range(args.iters_per_epoch), 
        print_freq=print_freq, 
        header=header):

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
        sample = sample.half().to("cuda:0")
    
        def check_for_nan(tensor, name):
          if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
          if torch.isinf(tensor).any():
            print(f"Inf detected in {name}")
          else:
            print(f"wow Tensor don't have NaN and Inf value")

        check_tensor = check_for_nan(tensor=sample, name="train_data")
        print(check_tensor)



        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=_dtype):
            rec_loss, gan_loss, log_loss = model(sample, args.global_step)

        ###################################################################################
        # THe update of rec_loss 
        if rec_loss is not None:
            loss_value = rec_loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value), force=True)
                sys.exit(1)


            with torch.autograd.set_detect_anomaly(True):

                optimizer.zero_grad()
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(rec_loss, optimizer, parameters=model.vae.parameters(), create_graph=is_second_order)

            # This loss_logger is used when datype is float16 or float32 not for the bfloat16
            if "scaler" in loss_scaler.state_dict():
                loss_scaler_value = loss_scaler.state_dict()["scaler"]
            else:
                loss_scaler_value = 1

            metric_logger.update(vae_loss=loss_value)
            metric_logger.update(loss_scaler=loss_scaler_value) # 1

        ############################################################################################

        # # The update of gan loss 
        if gan_loss is not None:
          gan_loss_value = gan_loss.item()

          if not math.isfinite(gan_loss_value):
                print("The gan discriminator Loss is {}, stopping training".format(gan_loss_value), force=True)
                sys.exit(1)

          with torch.autograd.set_detect_anomaly(True):
            optimizer_disc.zero_grad()
            is_second_order = hasattr(optimizer_disc, 'is_second_order') and optimizer_disc.is_second_order
            check_parameters = [param for param in model.loss.discriminator.parameters()]
            print(f"  ..........................what does discriminator parameters: {check_parameters}")
            check_tensor = check_for_nan(tensor=check_parameters[0], name="disc_param_data")
            print(check_tensor)
            # for name, param in model.loss.discriminator.parameters():
            #   print(weight), print(f"bias >>>>>>>>>>>>>>>> {bias}")
            # disc_grad_norm = loss_scaler_disc(gan_loss, optimizer_disc, parameters=model.loss.discriminator.parameters(), create_graph=True)
          
            
    #       if "scaler" in loss_scaler_disc.state_dict():
    #         disc_loss_scaler_value = loss_scaler_disc.state_dict()["scale"]
    #       else:
    #         disc_loss_scaler_value = 1 

    #       print(f"disc_loss_scaler_value: >>>>>>>> {disc_loss_scaler_value}")

    #       metric_logger.update(disc_loss=gan_loss_value)
    #       metric_logger.update(disc_loss_scale=disc_loss_scaler_value) # 1
    #       metric_logger.update(disc_grad_norm=disc_grad_norm) # 0.0

    #       min_lr = 10.
    #       max_lr = 0.
    #       for group in optimizer_disc.param_groups:
    #         min_lr = min(min_lr, group["lr"])
    #         min_lr = max(max_lr, group["lr"])

    #       metric_logger.update(lr=max_lr)
    #       metric_logger.update(min_lr=min_lr)
    #       weight_decay_value = None 

    #       for group in optimizer.param_groups:
    #         if group["weight_decay"] > 0:
    #           weight_decay_value = group["weight_decay"]

    #       metric_logger.update(weight_decay=weight_decay_value)
    #       metric_logger.update(grad_norm=grad_norm)

    #     torch.cuda.synchronize()
    #     new_log_loss = {
    #       k.split('/')[-1]: v
    #       for k, v in log_loss.items()
    #       if k not in ['total_loss']
    #     }
    #     metric_logger.update(**new_log_loss)

    #     if rec_loss is not None:
    #       min_lr = 10.
    #       max_lr = 0.
    #       for group in optimizer.param_groups:
    #         min_lr = min(min_lr, group["lr"])
    #         max_lr = max(max_lr, group["lr"])

    #       metric_logger.update(lr=max_lr)
    #       metric_logger.update(min_lr=min_lr)

    #       weight_decay_value = None
    #       for param in optimizer.param_groups:
    #         if group["weight_decay"] > 0:
    #           weight_decay_value = group["weight_decay"]

    #       metric_logger.update(weight_decay=weight_decay_value)
    #       metric_logger.update(grad_norm=grad_norm)

    #     if log_writer is not None:
    #       log_writer.update(**new_log_loss, head="train/loss")
    #       log_writer.update(lr=max_lr, head="opt")
    #       log_writer.update(min_lr=min_lr, head="opt")
    #       log_writer.update(weight_decay=weight_decay_value, head="opt")
    #       log_writer.update(grad_norm-grad_norm, head="opt")

    #       log_writer.set_step()

    #     args.global_step = args.global_step + 1 

    
    # print(f"Metric Logger: >>>>>>>>>>>>>>>>>>> : {metric_logger}")
    # return {
    #   k: meter.global_avg
    #   for k, meter in metric_logger.meters.items()
    # }
    

      



        






            
          
          

        

        





