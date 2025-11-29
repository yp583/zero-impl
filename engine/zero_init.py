import torch
import torch.nn as nn

class ZeroEngine:
    def __enter__(self):
        self.original_register = nn.Module.register_parameter
        def meta_register(self, name, param):
            if isinstance(param, nn.Parameter) and param.data.device.type != 'meta':
                meta = nn.Parameter(torch.empty_like(param, device='meta'),
                                    requires_grad=param.requires_grad)
                meta._shape_dtype = (tuple(param.shape), param.dtype)
                return self.original_register(self, name, meta)
            return self.original_register(self, name, param)
        
        nn.Module.register_parameter = meta_register 
        return None

    def __exit__(self, exc_type, exc, tb):
        nn.Module.register_parameter = self.original_register