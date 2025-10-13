
import torch 

from .metriclogger import MetricLogger, SmoothedValue

def train_epoch(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                optimizer_disc: torch.optim.Optimizer,
                epoch: int,
                print_freq: int = 20 ):
  
  # activate the training 
  model.train()

  metric_logger = MetricLogger()
  
  if optimizer is not None:
    metric_logger.add_meter(name="lr", meter=SmoothedValue(window_size=1, fmt='value:.6f'))
    metric_logger.add_meter(name="min_lr", meter=SmoothedValue(window_size=1, fmt='value:.6f'))

    

  if optimizer_disc is not None:
    metric_logger.add_meter(name="disc_lr", meter=SmoothedValue(window_size=1, fmt='value:.6f'))
    metric_logger.add_meter(name="disc_min_lr", meter=SmoothedValue(window_size=1, fmt='value:.6f'))

  header = f"Epoch: [{epoch}]"

  for step in metric_logger.log_every(iterable=range(2000), header=header, print_freq=print_freq):
    print(step)



