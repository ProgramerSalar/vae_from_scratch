import torch, time 
from collections import defaultdict, deque
from datetime import timedelta
from torch.cuda import is_available


class SmoothedValue(object):

  def __init__(self,
              window_size: int = 20,
              fmt: str = None):
    super().__init__()
    self.window_size = window_size 
    self.fmt = fmt 
    self.deque = deque(maxlen=window_size)

    self.total = 0.0 
    self.count = 0

  @property
  def value(self):
    return self.deque[-1]

  def update(self, value, n=1):
    self.deque.append(value)
    self.count += n 
    self.total += value * n 

  @property
  def avg(self):
    d = torch.tensor(list(self.deque), dtype=torch.float32)
    return d.mean().item()


  @property
  def global_avg(self):
    return self.total / self.count 




  def __str__(self):
    return self.fmt.format(
      value=self.value,
      avg=self.avg,
      global_avg=self.global_avg
    )





class MetricLogger(object):

  def __init__(self):
    super().__init__()
    self.meters = defaultdict(SmoothedValue)


  def update(self, **kwargs):

    for key, value in kwargs.items():
        if value is None:
            continue

        if isinstance(value, torch.Tensor):
            value = value.item()
        
        if isinstance(value, (float, int)):
          self.meters[key].update(value=value)

    return self.meters

  def log_every(self, iterable, header=None, print_freq=20):

    i = 0
    if not header:
      header = ''
    
    start_time = time.time()
    end_time = time.time()

    iteration_time = SmoothedValue(fmt='avg:.4f')
    data_time = SmoothedValue(fmt='{avg:.4f}')

    space_fmt = ':' + str(len(str(len(iterable)))) + 'd'       # :4d
    # log_msg = '[{0' + space_fmt + '}]/{1}'                     # [{0:4d}]/{1}
    log_msg = [
      header,
      '[{0' + space_fmt + '}]/{1}',
      'eta: {eta}',
    ]
    log_msg = str(log_msg)

    if torch.cuda.is_available():
      log_msg.append('GPU memory: {memory:.0f}')

    for obj in iterable:
      data_time.update(time.time() - end_time)
      yield obj 
      iteration_time.update(time.time() - end_time)

          # in the range of 20 iteration has been 0 
      if i % print_freq == 0 or \
          i == len(iterable) - 1:     # last iteration that is 1999
        
        eta_second = iteration_time.global_avg * (len(iterable) - 1)
        eta_string = str(timedelta(seconds=int(eta_second)))

        if torch.cuda.is_available():
          print(log_msg.format(
            i,
            len(iterable),
            eta=eta_string
          ))
        

      i += 1

    



            



  def add_meter(self, name, meter):
    # name = key, meter = value 
    self.meters[name] = meter 
    
    
  