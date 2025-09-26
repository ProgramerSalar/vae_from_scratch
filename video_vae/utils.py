import os, torch
from datetime import datetime, timedelta
import torch.distributed as distributed 

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
                "param": [],
                "lr": base_lr,
                "lr_scale": scale
            }

            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "param": [],
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








# if __name__ == "__main__":
    
#     out = init_distributed_mode(args=get_args)
