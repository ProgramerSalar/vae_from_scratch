import torch 
from torch import inf


class NativeScalerWithGradNormCount:


    """ 
        Mangaging Automatic Mixed Precision (AMP) training in pytorch. It wraps pyTorch native 
        `GrandScaler` to simplify the training loop, especially by adding gradient norm calculation and optional clipping.
    """

    # which is likely used when saving the training state (checkpointing) to identity 
    # the state  dictionary of the gradient scaler.
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        print(f"Set the loss scaled to {enabled}")

        # this helps prevent gradients from becoming zero ('underflow') 
        # when use lower-precision float point number (like float16). 
        # it does this by multiple the loss by a scaling factor before backpropagation 
        # and then unscaling the gradients before the optimizer updates the model weights.
        self._scaler = torch.amp.GradScaler(device="cuda",
                                            enabled=enabled)
        

    def __call__(self,
                 loss,
                 optimizer,
                 clip_grad=None,
                 parameters=None,
                 create_graph=False,
                 update_grad=True,
                 layer_names=None):
        
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None, "make sure parameters is not None"
                # it devide them by the scaling factor to bring them back to their original magnitude. 
                self._scaler.unscale_(optimizer=optimizer)
                # it modifies the gradient in place and return their norm before the clipping was applied.
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)

            else:
                self._scaler.unscale_(optimizer=optimizer)
                norm = get_grad_norm(parameters, layer_names=layer_names)


            self._scaler.step(optimizer)
            self._scaler.update()

        else:
            norm = None

        return norm
    

    def state_dict(self):
        return self._scaler.state_dict()
    

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


        

                

        
def get_grad_norm(parameters,
                  norm_type: float = 2.0,
                  layer_names=None) -> torch.Tensor:
    

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type), "make sure `norm_type` is float"
    if len(parameters) == 0:
        return torch.tensor(0,)

    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(
            p.grad.detach().abs().max().to(device)
            for p in parameters
        )

    else:
        layer_norm = torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters
            ])

        total_norm = torch.norm(layer_norm, norm_type)

        if layer_names is not None:
            if torch.isnan(total_norm) \
                or torch.isinf(total_norm) \
                or total_norm > 1.0:

                value_top, name_top = torch.topk(layer_norm, k=5)

                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

        return total_norm
            
