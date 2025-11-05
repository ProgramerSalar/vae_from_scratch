import torch 
from torch import inf, isnan
torch.autograd.set_detect_anomaly(True)




class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enable=True):
        print(f"Set the loss scaled to {enable}")
        self._scaler = torch.amp.GradScaler(device="cuda", enabled=enable)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)


        if update_grad:
          self._scaler.unscale_(optimizer)
          # torch.nn.utils.clip_grad_norm_(parameters, 1.0)
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
        
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    

    if len(parameters) == 0:
      return torch.tensor(0.)

    if torch.isnan(parameters[0]).any():
      print(f"........................................[Backward] NaN is not detected ")
    if torch.isinf(parameters[0]).any():
      print(f"........................................[Backward] Inf is not detected ")
    else:
      print(f"........................................[Backward] wow Tensor don't have NaN and Inf value")
    
    device = parameters[0].grad.device 
    layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
    total_norm = torch.norm(layer_norm, norm_type)
    
    return total_norm

    

