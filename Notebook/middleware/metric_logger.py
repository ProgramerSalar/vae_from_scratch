import torch, time
from collections import defaultdict, deque
from torch import distributed as dist
from datetime import timedelta




class MetricLogger(object):

    """
        tracking and logging various metrics (like loss, accuracy, time)
        during loop, typically a machine learning model it acts as a central hub 
        that holds multiple `SmoothedValue` objects, 

        each tracking a different metric. it's main feature in the `log_every` method 
        which wraps a loop and prints a nicely formatted, details progress bar with ETA (estimated time of arival)
        iteration times, and current metric values at regular interval.
    """

    def __init__(self,
                 delimiter="\t"):
        
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):

        """This method is used to update the values of one or more metrics."""

        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            
            assert isinstance(v, (float, int)), "make sure value is `float` or `int`"
            self.meters[k].update(v)    # update the key value 

    def __getattr__(self, attr):

        """This is a specail python method that gets called when you try to access an attribute 
            that doesn't exit in the usual way. (e.g, `logger.loss`)"""

        if attr in self.meters:
            return self.meters[attr]
        
        if attr in self.__dict__:
            return self.__dict__[attr]
        
        raise AttributeError(f"{type(self).__name__} object has not attributes {attr}")
    

    def __str__(self):

        """This special method defines what happens when you try to convert the logger object 
            to a string, for example by using `print(logger)`"""

        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f"{name}: {str(meter)}"
            )
        return self.delimiter.join(loss_str)
    
    
    def synchronize_between_processes(self):

        """This is helper method for distributed training."""

        for meter in self.meters.values():
            meter.synchronize_between_process()


    def add_meter(self, name, meter):
        self.meters[name] = meter

    
    def log_every(self,
                  iterable, # the data source to loop over (e.g trainig data)
                  print_freq,   # how often (in number of iterations) to print a log message.
                  header=None   # An optional string to print at the start of epoch log line (e.g., "Epoch: [1]")
                  ):
        
        i = 0 
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        # creates a format specifier for padding the iteration count. 
        # for example if `iterable` has 8000 items `len(str(len(iterable)))` is 4
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # Epoch: [1] [9/8000] eta: 0:01:59 ...
        # Epoch: [1] [10/8000] eta: 0:01:58 ...
        # Epoch: [1] [999/8000] eta: 0:01:21 ...
        # Epoch: [1] [1000/8000] eta: 0:01:20 ...
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]"
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}"
        ]

        # gpu memory reading...
        if torch.cuda.is_available():
            log_msg.append('max_mem: {memory:.0f}')

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            # measure the time spent just loading the data.
            data_time.update(time.time() - end)
            # This is what makes the method a generator. it yields the data item `obj` back to the calling code (e.g., the training loop)
            # which then performs its operations. After the calling code is done, excecution returns here.
            yield obj
            # measure the total time for the whole iteration (data loading + processing)
            iter_time.update(time.time() - end)
            

            # checks if it's time to print a log: either the iteration number is a multiple of `print_freq` or it's the very last iteration.
            if i % print_freq == 0 or i == len(iterable) - 1:

                # calculate the eta (Estimated time of Arrival) by multiplying the average iteration time by the number of 
                # remaining iterations. it then formats this into a human-readable `HH:MM:SS`
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))

                else:
                    print(log_msg.format(
                        i, 
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=self(iter_time),
                        data=str(data_time)
                    ))

            # increments the iteration counter and resets the `end` timer for the next loop.
            i += 1 
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))

        # final summary showing the total_time taken and the average time per item over the entire run.
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f})")





        







        






class SmoothedValue(object):

    """ 
        A utility for tracking metrics during a process like training a machine learning model.
        it keeps track of the most recent values in a fixed-size "window" as well as the global average of all values seen.

        it is partcularly useful for logging training progress, where you might want to see both the recent performance 
        (e.g., over the last 20 batches) and the overall performance since the begining.
    """

    def __init__(self,
                 window_size=20, # max num of recent values to keep track
                 fmt=None   # format string to control how the object's value is displayed when printed.
                 ):
        

        if fmt is None:
            # it's configured to show the median of the window and the global average,
            # both formatted to four decimal places.
            fmt = "{median:.4f} ({global_avg:.4f})"

        # once it's full add a new element automatically discards the older element.
        self.deque = deque(maxlen=window_size)
        self.total = 0.0 
        self.count = 0 
        self.fmt = fmt 

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n 
        self.total += value * n 

    def synchronize_between_process(self):
        """
            This method to disigned for distributed training (running code on multiple GPUs or machines).
            it's purpose is to combine the `total` and `count` from all process to get a single correct global average.
        """

        if not is_dist_avail_and_initialized():
            return 
        
        t = torch.tensor([self.count, self.total], 
                         dtype=torch.float64,
                         device='cuda')
        

        # Pauses each process until all other process have reached this point. 
        # This prevents errors for process running at different speeds.
        dist.barrier()
        # This is the key synchronization step. It sums the tensor `t` from all 
        # process and distributed the result back to `t` in every process.
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """This method calculate and return the median of the values currently  in the `deque` window."""

        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque),
                         dtype=torch.float32)
        return d.mean().item()
    

    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        """This property returns the most recent value that was added to the `deque`"""
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )
    



        
def is_dist_avail_and_initialized():

    if not dist.is_available():
        return False
    
    if not dist.is_initialized():
        return False
    
    return True







