import torch 
import json
import apex
has_apex=True

def create_optimizer(args,
                     model,
                     get_num_layer=None,
                     get_layer_scale=None,
                     filter_bias_and_bn=True,
                     skip_list=None,
                     **kwargs):
    
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay 

    skip = {}   # default value 
    if skip_list is not None:
        skip = skip_list

    # weight decay is a `regularization technique` used to prevent a model from overfitting. 
    # it works by adding a small penality to the loss function for large parameter values.
    # The encourages the optimizer to keep the models' weights small and therfore less complex. 
    # it's common and effective way to improve a model's ability to generalize a new data.
    elif hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    print(f"Skip weight decay name marked in model: {skip}")

    parameters = get_parameter_groups(model=model,
                                      weight_decay=weight_decay,
                                      base_lr=args.lr,
                                      skip_list=skip,
                                      get_num_layer=get_num_layer,
                                      get_layer_scale=get_layer_scale,
                                      **kwargs)
    

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), "Apex and Cuda required for fused optimizers."

    opt_args = dict(lr=args.lr,
                    weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps 

    if hasattr(args, 'opt_beta1') and args.opt_beta1 is not None:
        opt_args['beta'] = (args.opt_beta1, args.opt_beta2)

    print(f"Optimizer config: {opt_args}")
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    # optimizer 
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = torch.optim.SGD(params=parameters,
                                    momentum=args.momentum, # but momentum does not found in the argument
                                    nesterov=True,
                                    **opt_args)
        
    if opt_lower == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args)

    if opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)

    if opt_lower == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, **opt_args)

    else:
        print(f"Optimizer: {opt_lower}")
        raise ValueError
    
    return optimizer



    


    



def get_parameter_groups(model,
                         weight_decay=1e-5,
                         base_lr=1e-4,
                         skip_list=(),
                         get_num_layer=None,
                         get_layer_scale=None,
                         **kwargs):
    

    """ 
        This function is designed to create fine-grained parameter groups for a pytorch 
        optimizer. This allows you to apply different setting (like learning_rates or weight_decay)
        to different part of a model.
    """

    

    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue    # frozen weight

        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                # matching case 
                # filter_name = 'bias'
                # name = 'encoder.layer.attention.output.dense.bias'
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True

            if flag:
                continue

        

        # param= torch.Tensor()
        # name='encoder.layer.attention.output.dense.bias' 
        # skip_list = (['encoder.layer.attention.output.dense.bias', .....])
        if param.ndim <= 1 or \
            name.endswith('.bias') or name in skip_list:

            group_name = "no_decay"
            this_weight_decay = 0. 

        else:
            group_name = "decay"
            this_weight_decay = weight_decay


        # layer-wise grouping 
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = f"layer_{layer_id}_{group_name}"

        else:
            layer_id = None

        parameter_group_names = {}
        parameter_group_vars = {}
        default_scale = 1.
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

        parameter_group_names[group_name]["params"].append(param)
        parameter_group_vars[group_name].append(name)

    print(f"Pram groups: {json.dumps(parameter_group_names, indent=2)}")
    return list(parameter_group_vars.values())














