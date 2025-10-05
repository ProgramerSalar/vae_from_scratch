import torch, time
from collections import defaultdict, deque
from datetime import timedelta


class MetricLogger(object):

    """ 
        tracking and logging various metrics (like., loss, accuracy, time)
        during loop

        Args:
            delimiter: the space of your print style.
    """

    def __init__(self,
                 delimiter="\t"):
        
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter


    def update(self, **kwargs):

        """This function update the value tensor."""

        for k, v in kwargs.items():
            if v is None:
                continue

            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (float, int)), "make sure value is `float` or `int`"
            self.meters[k].update(v)


    def __getattr__(self, attr):

        """this method called when you try to access in attribute that doen't exit in the usual way. (e.g, `logger.loss`)"""

        if attr in self.meters:
            return self.meters[attr]
        
        if attr in self.__dict__:
            return self.__dict__[attr]
        

        raise AttributeError(F"{type(self).__name__} object has no attributes {attr}")
    

    def __str__(self):

        """Convert the value into the string format."""

        loss_str = []
        for key, value in self.meters.items():
            
            loss_str.append(
                f'{key}: {str(value)}'
            )

        result = self.delimiter.join(loss_str)
        return result
    

    def synchronize_between_processes(self):

        """
            This method to disigned for distributed training (running code on multiple GPUs or machines).
            It's purpose is to combine the `total` and `count` from all process to get a single correct global average.
        """

        for value in self.meters.values():
            value.synchronize_between_process()


    def add_meter(self,
                  name, 
                  meter):
        
        """add the value of the name."""

        self.meters[name] = meter


    def log_ever(self,
                 iterable,
                 print_freq,
                 header=None):
        
        """
            Arrange the print text.

            iterable: the data souce to loop over (e.g., training data)
            print_preq: how often (in number of iterations) to print a log message.
            header: An optional string to print at the start of epoch log line (e.g., "Epoch: [1]")
        """
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end_time = time.time()

        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        # creates a format specifier for padding the iteration count. 
        # for example if `iterable` has 8000 items `len(str(len(iterable)))` is 4
        space_format = ':' + str(len(str(len(iterable)))) + 'd'


        log_msg = [
            header,
            "[{0" + space_format + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}"
        ]

        # gpu memory reading...
        if torch.cuda.is_available():
            log_msg.append("max_memory: {momery:.0f}")

        # add the space to log_msg
        log_msg = self.delimiter.join(log_msg)

        for obj in iterable:

            data_time.update(time.time() - end_time)
            # This is what makes the method a generator. it yields the data item `obj` back to the calling code (e.g., the training loop)
            # which then performs its operations. After the calling code is done, excecution returns here.
            yield obj

            iter_time.update(time.time() - end_time)

            if i % print_freq == 0 or i == len(iterable) - 1:
                
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=self(iter_time),
                        data=str(data_time)
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

            i += 1 
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))

        print(f"{header} Total time: {total_time_str}, {total_time / len(iterable):.4f}")

            



class SmoothedValue(object):


    """ 
        A utility function to track metrics during a process like training a machine learning model.
        it keeps track of the most rescent values in the fixed-size "window" as well as the global average of all values seen.

        it is particularly useful for logging training process, where you might want to see both the recent performance.
        (e.g., over the last 20 batches) and the overall performance since the begining.

        Args:
            window_size: max mumber of recent values to keep track.
            fmt: format string to control how to object's value is displayed when printed.
    """

    def __init__(self,
                 window_size: int =20,
                 fmt: str = None):
        
        assert window_size is not None, "make sure `window_size` is not None."
        assert fmt is not None, "make sure `fmt` is not None."

        self.deque = deque(maxlen=window_size)
        self.count = 0
        self.total = 0.0
        self.fmt = fmt
        self.window_size = window_size

        
    def update(self, 
               value,
               n=1):
        
        """update the value."""
        
        self.deque.append(value)
        self.count += n 
        self.total += value * n 


    def synchronize_between_process(self):

        """This method to disigned for distributed training (running code on multiple GPUs or machines).
            It's purpose is to combine the `total` and `count` from all process to get a single correct global average.
        """

        assert check_init_dist_avail_and_initialized, SystemError

        # Pause each process unitl all other process have reached this point.
        # This prevents errors for process running at different speeds.
        torch.distributed.barrier()

        tensor = torch.tensor([self.count, self.total],
                              dtype=torch.float64,
                              device='cuda')

        # it sums to tensor from all process and distributed the result back to tensor in every process.
        torch.distributed.all_reduce()
        tensor = tensor.tolist()
        self.count = int(tensor[0])
        self.total =  tensor[1]


    @property
    def median(self):
        """ This method calculate the median of the tensor."""

        median_tensor = torch.tensor(self.deque)
        median_tensor = median_tensor.median().item()

        return median_tensor
    
    @property
    def mean(self):
        """This method to calculate the mean of the tensor."""

        mean_tensor = torch.tensor(self.deque,
                                   dtype=torch.float32)
        
        mean_tensor = mean_tensor.mean().item()

        return mean_tensor
    

    @property
    def global_avg(self):
        return self.total / self.count
    

    @property
    def value(self):
        """The last value to extract the deque list."""
        return self.deque[-1]
    

    @property
    def max(self):
        return max(self.deque)
    

    def __str__(self):

        """This function  help to calculate the format of the value."""

        format_string = self.fmt.format(
            median=self.median,
            avg=self.mean,
            global_avg=self.global_avg,
            max_value=self.max,
            value=self.value

        )
        



def check_init_dist_avail_and_initialized():

    """
        Check the distributed backend (like NCCL, Gloo or MPI) is available on the system.
        and distributed process group has been properly initialized for communication between different process.
    """

    if not torch.distributed.is_available():
        return False
    
    if not torch.distributed.is_initialized():
        return False
    
    return True




    





        
        





    
