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
                print(grad_norm)

            
     

        



        






            
          
          

        

        





