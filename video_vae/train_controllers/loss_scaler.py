import torch


class LossScaler:

    """ 
        Manage Automatic mixed precision (AMP) training. 
        
        Args:
            enable (bool): do you want to  Manage the AMP training.
    """

    def __init__(self,
                 enabled=True,
                 ):
        
        print(f"Manage AMP loss_scaler: {enabled}")

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
                 ):
        

        """ 
            Args:
                loss: model loss. 
                optimizer: optimizer. 
                clip_grad: the normalization value. 
                parameters: model parameters. 
                create_graph: loss graph is create or not. 
                update_grad: model gradient is update or not.
        """

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # scaled_loss = original_loss x loss_scale_factor [3e-6 x 65536 => 0.0196608] 
        # scaled_gradient = original_gradient x loss_scale_facter
        self._scaler.scale(loss).backward()

        if update_grad:
            # if clip_grad is not None:
            #     assert parameters is not None, "make sure parameter is not None!"

            #     # it divide them by the scale factor to bring them back to their original value.
            #     # unscale_gradient = scale_gradient / loss_scale_factor 
            #     self._scaler.unscale_(optimizer=optimizer)

            #     # if modifies the gradient in place and return their norm before the clipping was applied.
            #     norm = torch.nn.utils.clip_grad_norm_(parameters=parameters,
            #                                           max_norm=clip_grad)
                

            # else:
            #     print('work in progress...')
                
            
            self._scaler.step(optimizer=optimizer)
            self._scaler.update()


        else:
            norm = None



        return norm
            







        

