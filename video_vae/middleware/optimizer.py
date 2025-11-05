import torch, json
import torch.optim as optim



def create_optimizer(args,
                    model,
                     **kwargs):
    
    
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay 
    lr=args.lr
    parameters = get_parameter_groups(model=model, weight_decay=weight_decay, base_lr=lr)
    
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps'):
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas'):
        opt_args['betas'] = (args.opt_betas)

    # print(opt_args)

    if opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(params=parameters,
                                      **opt_args)
        
    return optimizer







def get_parameter_groups(model, weight_decay=1e-5, base_lr=1e-4):
    
    parameter_group_name = {}
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        
        
        if not param.requires_grad:
            continue    # frozen weight 
        
        if param.ndim <=1 or name.endswith('.bias'):
                
            group_name = "no_decay"
            this_weight_decay = 0. 

        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        scale = 1.
        if group_name not in parameter_group_name:
            parameter_group_name[group_name] = {
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
        
        parameter_group_name[group_name]["params"].append(name)
        parameter_group_vars[group_name]["params"].append(param)

    # print("Param group = {json.dumps(parameter_group_name, indent=2)}")
    return list(parameter_group_vars.values())

    



        
            


