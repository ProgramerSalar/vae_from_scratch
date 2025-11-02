import torch 



class NativeScalerWithGradNormCount:

    def __init__(self, enable=True):
        self._scaler = torch.amp.GradScaler(device="cuda", enabled=enable)

    def __call__(self,
                 loss,
                 optimizer,
                 parameters,
                 create_graph=False,   # optimizer second_order
                 retain_graph=False,
                 update_grad=True,
                 layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            self._scaler.unscale_(optimizer)
            norm = get_grad_norm(parameters=parameters, layer_names=layer_names)
            
        self._scaler.step(optimizer)
        self._scaler.update()

        return norm
    
    def state_dict(self):
        return self._scaler.state_dict()
    

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

        




def get_grad_norm(parameters,
                  norm_type=2.0,
                  layer_names=None):

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
        print(f"[native_scaler] Really paramaters is Tensor: >>>>>>>>>>>>>>>>> {parameters}")

    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
     
    if len(parameters) == 0:
        return torch.tensor(0.)
    
    device = parameters[0].grad.device 

    layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
    total_norm = torch.norm(layer_norm, norm_type)

    if layer_names is not None:
        if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
            value_top, name_top = torch.topk(layer_norm, k=5)
            print(f"Top norm value: {value_top}")
            print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

    return total_norm

    

