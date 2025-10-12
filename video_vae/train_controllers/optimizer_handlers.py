import torch 


def Optimizer_handler(model):

  model_parameters = model.parameters()
  optimizer = torch.optim.AdamW(params= model_parameters    # model parameters 
                                )

  return optimizer