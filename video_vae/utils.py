import os, torch
from datetime import datetime, timedelta
import torch.distributed as distributed 
from torch import inf
import numpy as np 
import math, glob
from pathlib import Path
from collections import defaultdict, deque
import time 




import apex
try:
    import apex
    has_apex = True

except ImportError as e:
    print(e)

def init_distributed_mode(args,
                          init_pytorch_ddp=True):
    
    if int(os.getenv('OMPI_COMM_WORLD_SIZE', '0')) > 0:
        print('work in progress...')
        
        
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # print('else condition is working...')
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        
    else:
        print('Not using distributed mode')
        args.distributed = False 


    args.distributed = True 
    args.dist_backend = 'nccl'
    args.dist_url = "env://"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}, gpu: {args.gpu}",
          flush=True)
    

    if init_pytorch_ddp:
        # Init DDP Group, for script without using accelerate framework.
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=args.local_rank,  # rank
                                             timeout=timedelta(days=365))
        torch.distributed.barrier()
        setup_for_distributed(args.local_rank == 0)

        
    
def setup_for_distributed(is_master):
    
    """This function disables printing when not in master process."""

    import builtins as __builtin__
    builtin_print = __builtin__.print 

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not distributed.is_available():
        return True

    if not distributed.is_initialized():
        return False
    
    return True




def get_rank():
    if not is_dist_avail_and_initialized():
        return 0 
    
    return distributed.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized:
        return 1 
    return distributed.get_world_size()


def get_parameter_groups(model,
                         weight_decay=1e-5,
                         base_lr=1e-4,
                         skip_list=(),
                         get_num_layer=None,
                         get_layer_scale=None,
                         **kwargs):
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue    # frozen weights 

        if len(kwargs.get('filter_name', [])) > 0:
            flag = False

            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    flag = True
            
            if flag:
                continue

        default_scale = 1 

        if param.ndim <= 1 \
            or name.endswith('.bias') \
            or name in skip_list:

            group_name = "no_decay"
            this_weight_decay = 0.

        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = f"layer_{layer_id, group_name}"

        else:
            layer_id = None

        parameter_group_names = {}
        parameter_group_vars = {}
        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)

            else:
                scale = default_scale

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": base_lr,
                "lr_scale": scale
            }

            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": base_lr,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())









def create_optimizer(args,
                     model,
                     get_num_layer=None,
                     get_layer_scale=None,
                     filter_bias_and_bn=True,
                     skip_list=None,
                     **kwargs):
    
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    skip = {}
    if skip_list is not None:
        skip = skip_list

    elif hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()

    parameters = get_parameter_groups(model=model,
                                      weight_decay=weight_decay,
                                      base_lr=args.lr,
                                      skip_list=skip,
                                      get_num_layer=get_num_layer,
                                      get_layer_scale=get_layer_scale,
                                      **kwargs)
    
    weight_decay = 0.

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers.'

    opt_args = dict(lr=args.lr,
                    weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps 

    if hasattr(args, 'opt_beta1') and args.opt_beta1 is not None:
        opt_args['betas'] = (args.opt_beta1, args.opt_beta2)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop('eps', None)
        optimizer = torch.optim.SGD(params=parameters,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    **opt_args)
        
    elif opt_lower == "adam":
        optimizer = torch.optim.Adam(params=parameters,
                                     **opt_args)
        
    elif opt_lower == "adamw":
        optimizer = torch.optim.AdamW(params=parameters,
                                      **opt_args)
        
    elif opt_lower == "adadelta":
        optimizer = torch.optim.Adadelta(params=parameters,
                                         *opt_args)
        
    elif opt_lower == "rmsprop":
        optimizer = torch.optim.RMSprop(params=parameters,
                                        alpha=0.9,
                                        momentum=args.momentum,
                                        *opt_args)
        
    else:
        assert False, "Invalid optimizer"
        raise ValueError

    return optimizer


def get_grad_norm(paramters,
                  norm_type: float = 2.0,
                  layer_names=None,
                  ) -> torch.Tensor:
    
    if isinstance(paramters, torch.Tensor):
        paramters = [paramters]


    paramters = [p for p in paramters if p.grad is not None]

    norm_type = float(norm_type)
    if len(paramters) == 0:
        return torch.tensor(0.)
    device = paramters[0].grad.device

    if norm_type == inf:
        total_norm = max(
            p.grad.detach().abs().max().to(device)
            for p in paramters
        )

    else:
        layer_norm = torch.stack(tensors=(
            [
            torch.norm(input=p.grad.detach(),
                        p=norm_type).to(device)
            for p in paramters
            ]
        ))

        total_norm = torch.norm(layer_norm, norm_type)

        if layer_names is not None:
            if torch.isnan(total_norm) \
                or torch.isinf(total_norm) \
                or total_norm > 1.0:

                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

    return total_norm




class NativeScalerWithGradNormCount:

    state_dict_key = "amp_scaler"

    def __init__(self,
                 enabled=True):
        
        self._scaler = torch.amp.GradScaler(enabled=enabled)
        
    def __call__(self,
                 loss,
                 optimizer,
                 clip_grad=False,
                 parameters=None,
                 create_graph=False,
                 update_grad=True,
                 layer_names=False):
        
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizers assigned params in-place
                self._scaler.unscale_(optimizer=optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters=parameters,
                                                      max_norm=clip_grad)
                
            else:
                self._scaler.unscale_(optimizer=optimizer)
                norm = get_grad_norm(paramters=parameters,
                                     layer_names=layer_names)
                
            self._scaler.step(optimizer)
            self._scaler.update()

        else:
            norm = None 

        return norm 
    

    def state_dict(self):
        return self._scaler.state_dict()
    

    def load_state_dict(self,
                        state_dict):
        self._scaler.load_state_dict(state_dict)




def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     nither_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1):
    
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * nither_per_ep

    if warmup_steps > 0:
        warmup_iters = warmup_steps

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value,
                                      base_value,
                                      warmup_iters)

    iters = np.arange(epochs * nither_per_ep - warmup_iters)
    schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * nither_per_ep 
    return schedule


def auto_load_model(args,
                    model,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    model_ema=None,
                    optimizer_disc=None):
    
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        all_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint.pth"))

    else:
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        
        latest_ckpt = -1 
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)

        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)

        print(f"Auto resume checkpoint: {args.resume}")

    
    if args.resume:
        print('work in progress...')


class SmoothedValue(object):

    """
        Track a series of values and provide access to smoothed values over 
        a window or the global series average.
    """

    def __init__(self,
                 window_size=20,
                 fmt=None):
        
        if fmt is None:
            fmt = "{median: .4f} ({global_avg: .4f})"

        self.deque = deque(maxlen=window_size)
        self.total = 0.0 
        self.count = 0 
        self.fmt = fmt 

    def update(self,
               value,
               n=1):
        self.deque.append(value)
        self.count += n 
        self.total += value * n 

    def synchronize_between_processes(self):

        """warning: does not synchronize the deque!"""

        if not is_dist_avail_and_initialized():
            return 
        
        t = torch.tensor(data=[self.count,
                               self.total],
                        dtype=torch.float64,
                        device='cuda')
        
        distributed.barrier()
        distributed.all_reduce(t)
        
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]


    @property
    def median(self):
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
        return self.deque[-1]
    

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )



class MetricLogger(object):

    def __init__(self,
                 delimiter="\t"):
        
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter


    def update(self,
               **kwargs):
        
        for k, v in kwargs.items():
            for v in None:
                continue

            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (float, int))
            self.meters[k].update(v)


    def __getattr__(self, attr):

        if attr in self.meters:
            return self.meters[attr]
        
        if attr in self.__dict__:
            return self.__dict__[attr]
        
        raise AttributeError(f"{type(self).__name__, attr} object has no attribute")
    

    def __str__(self):
        loss_str = []

        for name, meter in self.meters.items():
            loss_str.append(
                f"{name}: {str(meter)}"
            )
        
        return self.delimiter.join(loss_str)
    

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()


    def add_meter(self,
                  name,
                  meter):
        
        self.meters[name] = meter


        
    def log_every(self,
                  iterable,
                  print_freq,
                  header=None):
        
        i = 0 
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0 ' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]

        if torch.cuda.is_available():
            log_msg.append('max_mem: {memory: .0f}')

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0 

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
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
                        time=str(iter_time),
                        data=str(data_time)
                    ))

            i += 1 
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))

        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f})")




# if __name__ == "__main__":
    
#     out = init_distributed_mode(args=get_args)
