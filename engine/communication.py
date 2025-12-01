def gather_params_for_forward(rank: int, module):
    # check whether this rank has params associated with module if so broadcast them
        # ways to do this
        # 1. have a map (rank -> list(params)) and loop over params to see if they belong to this module
        # 2. have a map (param -> rank) loop over params and get the rank and check against this rank to decide whether to recv or broadcast
        # 3. add a seperate param not on meta that will have a materialized_ prefix and check if the module has it for the modules named params

    # recieve the params for the module from other ranks that dont have materialized version

    for name, _ in module.named_parameters():
        materialized_param = getattr(module, f"materialized_{name}", None)
        if materialized_param is not None:
            #naive to just send need ot bucket 
            pass


def discard_params_after_forward(module):
    # discard all params besides the ones for this rank
    pass

def gather_grads_for_backward(module):
    pass

def discard_grads_after_backward(module):
    
    pass
