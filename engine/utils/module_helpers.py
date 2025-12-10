import torch.nn as nn

def has_direct_params(module: nn.Module) -> bool:
    return next(module.parameters(recurse=False), None) is not None

def get_module_tree(module: nn.Module, include_params: bool = False) -> dict[str, list[str]]:
    adj_list = {}

    def build_tree(mod: nn.Module, parent_name: str = ""):
        name = parent_name or mod.__class__.__name__
        children = []

        for child_name, child_mod in mod.named_children():
            full_name = f"{name}.{child_name}"
            children.append(full_name)
            build_tree(child_mod, full_name)

        if include_params:
            for param_name, _ in mod.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}"
                children.append(full_name)
                adj_list[full_name] = []

        adj_list[name] = children

    build_tree(module)
    return adj_list

def get_shard_numels(model: nn.Module):
    model_numels = {}
    for name, param in model.named_parameters():
        sharded_param = getattr(param, "_shard_state", None)
        shard_numel = sharded_param.materialized.numel() if sharded_param is not None and sharded_param.materialized is not None else 0
        full_numel = param.numel()
        model_numels[name] = f"{shard_numel} / {full_numel}"
    return model_numels
