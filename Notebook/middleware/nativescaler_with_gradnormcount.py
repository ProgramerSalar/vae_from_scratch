import torch 



class NativeScalerWithGradNormCount:

    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        print(f"Set the loss scaled to {enabled}")

        self._scaler = torch.amp.grad_scaler()