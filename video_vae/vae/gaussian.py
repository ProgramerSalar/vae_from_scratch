import torch 
from torch import nn 
from typing import Tuple, Union, Optional, List
import numpy as np 
# from diffusers.utils.torch_utils import randn_tensor

class DiagonalGaussianDistribution(object):

    def __init__(self,
                 parameters: torch.Tensor,
                 deterministic: bool = False):
        
        self.parameters = parameters
        self.deterministic = deterministic
        self.mean, self.logvar = torch.chunk(input=parameters,
                                             chunks=2,
                                             dim=1)
        
        self.logvar = torch.clamp(input=self.logvar,
                                  min=-30.0,
                                  max=20.0)
        
        self.std = torch.exp(input=0.5 * self.logvar)
        self.var = torch.exp(input=self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                input=self.mean,
                device=self.parameters.device,
                dtype=self.parameters.dtype
            )

    def sample(self,
               generator: torch.Generator = None) -> torch.FloatTensor:
        
        # sample = randn_tensor(shape=self.mean.shape,
        #              generator=generator,
        #              device=self.parameters.device,
        #              dtype=self.parameters.dtype)

        sample = torch.randn(size=self.mean.shape,
                             generator=generator,
                             device=self.parameters.device,
                             dtype=self.parameters.dtype)
        
        x = self.mean + self.std * sample
        return x 
    


    def kl(self,
           other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        
        if self.deterministic:
            return torch.Tensor([0.0])
        
        else:
            if other is None:
                return 0.5  * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[2, 3, 4]
                )
        
            else:
                return 0.5 * torch.sum(
                    input=torch.pow(self.mean - other.mean, 2) / other.var 
                    + self.var / other.var 
                    - 1.0 
                    - self.logvar
                    + other.logvar,
                    dim=[2, 3, 4]
                )
            

    def nll(self,
            sample: torch.Tensor,
            dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        
        if self.deterministic:
            return torch.Tensor([0.0])
        
        logtwopi = np.log(2.0 * np.pi)

        result = 0.5 * torch.sum(
            logtwopi + self.logvar 
            + torch.pow(sample - self.mean, 2) / self.var ,
            dim=dims
        )


    def mode(self) -> torch.Tensor:
        return self.mean



if __name__ == "__main__":

    tensor = torch.randn(2, 6, 1, 32, 32)
    model = DiagonalGaussianDistribution(parameters=tensor)
    print(model.mode().shape)

