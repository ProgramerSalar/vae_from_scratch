import torch 


def Optimizer_handler(model):

  model_parameters = model.parameters()
  weight_decay = 1e-4
  lr = 5e-5
  eps = 1e-8
  betas = (0.9, 0.95)

  opt_args = dict(lr=lr, weight_decay=weight_decay, eps=eps, betas=betas)
  optimizer = torch.optim.AdamW(params= model_parameters,    # model parameters 
                                **opt_args)

  return optimizer