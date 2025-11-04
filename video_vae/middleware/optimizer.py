import torch, json
import torch.optim as optim

def get_parameter_groups(model, weight_decay=1e-5, base_lr=1e-4, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue

        default_scale=1.
        
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = default_scale

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": base_lr,
                "lr_scale": scale,
            }

            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": base_lr,
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    skip = {}
    if skip_list is not None:
        skip = skip_list
    elif hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    print(f"Skip weight decay name marked in model: {skip}")
    parameters = get_parameter_groups(model, weight_decay, args.lr, skip, get_num_layer, get_layer_scale, **kwargs)
    weight_decay = 0.

   
    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = (args.opt_betas)
    
    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


###############################################################

# def create_optimizer(args,
#                     model,
#                      **kwargs):
    
    
#     opt_lower = args.opt.lower()
#     weight_decay = args.weight_decay 
#     lr=args.lr
#     parameters = get_parameter_groups(model=model, weight_decay=weight_decay, base_lr=lr)
    
#     opt_args = dict(lr=lr, weight_decay=weight_decay)
#     if hasattr(args, 'opt_eps'):
#         opt_args['eps'] = args.opt_eps
#     if hasattr(args, 'opt_betas'):
#         opt_args['betas'] = (args.opt_betas)

#     # print(opt_args)

#     if opt_lower == 'adamw':
#         optimizer = torch.optim.AdamW(params=parameters,
#                                       **opt_args)
        
#     return optimizer







# def get_parameter_groups(model, weight_decay=1e-5, base_lr=1e-4):
    
#     parameter_group_name = {}
#     parameter_group_vars = {}
#     for name, param in model.named_parameters():
        
        
#         if not param.requires_grad:
#             continue    # frozen weight 
        
#         if param.ndim <=1 or name.endswith('.bias'):
                
#             group_name = "no_decay"
#             this_weight_decay = 0. 

#         else:
#             group_name = "decay"
#             this_weight_decay = weight_decay

#         scale = 1.
#         if group_name not in parameter_group_name:
#             parameter_group_name[group_name] = {
#                 "weight_decay": this_weight_decay,
#                 "params": [],
#                 "lr": base_lr,
#                 "lr_scale": scale
#             }
#             parameter_group_vars[group_name] = {
#                 "weight_decay": this_weight_decay,
#                 "params": [],
#                 "lr": base_lr,
#                 "lr_scale": scale
#             }
        
#         parameter_group_name[group_name]["params"].append(name)
#         parameter_group_vars[group_name]["params"].append(param)

#     # print("Param group = {json.dumps(parameter_group_name, indent=2)}")
#     return list(parameter_group_vars.values())

    



        
            


