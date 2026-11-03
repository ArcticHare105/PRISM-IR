"""Check weight loading and phase correctness; no model-size or speed statistics."""
import copy
import torch


def model_structure(model):
    return {
        "ranks": {f"rank_{network}_{adapter}": getattr(model, f"lora_rank_{network}_{adapter}")
                  for network in ("unet", "vae") for adapter in ("deg", "vq")},
        "selector_dimensions": {"unet": [1024, model.rank_unet * 4, model.rank_unet * 2, model.rank_unet],
                                "vae": [1024, model.rank_vae * 4, model.rank_vae * 2, model.rank_vae]},
        "adapters": {network: {name: {"rank": config.r, "alpha": config.lora_alpha}
                               for name, config in getattr(model, network).peft_config.items()}
                     for network in ("unet", "vae")},
        "adapter_rebase": copy.deepcopy(model.adapter_rebase), "timestep": int(model.timesteps.item())}


def check_checkpoint(model, checkpoint):
    expected = torch.load(checkpoint, map_location="cpu")
    actual = model.checkpoint_state()
    if set(expected) != set(actual):
        raise ValueError("Checkpoint/model top-level fields differ")
    for field, value in expected.items():
        if field.startswith("state_dict_"):
            loaded = actual[field]
            if set(value) != set(loaded):
                raise ValueError(f"Checkpoint/model tensor names differ: {field}")
            for name, tensor in value.items():
                if tensor.shape != loaded[name].shape or not torch.equal(tensor, loaded[name]):
                    raise ValueError(f"Checkpoint/model tensor mismatch: {field}.{name}")
        elif actual[field] != value:
            raise ValueError(f"Checkpoint/model metadata differs: {field}")
    return check_training_phases(model)


def check_fresh_initialization(model):
    for network in (model.unet, model.vae):
        for name, parameter in network.named_parameters():
            if "lora_B." in name and torch.any(parameter.detach() != 0):
                raise ValueError(f"Fresh LoRA B must be zero: {name}")
            if "lora_A." in name and (not torch.isfinite(parameter).all() or not torch.any(parameter.detach() != 0)):
                raise ValueError(f"Fresh LoRA A must have finite random weights: {name}")
    for name, parameter in model.vae.decoder.named_parameters():
        if name.startswith("skip_conv_") and not torch.all(parameter.detach() == 1e-5):
            raise ValueError(f"Unexpected fresh skip-convolution initialization: {name}")
    return check_training_phases(model)


def check_training_phases(model):
    for phase in ("deg", "vq"):
        model.set_phase(phase)
        other = "vq" if phase == "deg" else "deg"
        if any(f"_{other}." in name and p.requires_grad for name, p in model.named_parameters()):
            raise ValueError(f"Inactive LoRA branch is trainable in {phase} phase")
        if not any(f"_{phase}." in name and p.requires_grad for name, p in model.named_parameters()):
            raise ValueError(f"Active LoRA branch is frozen in {phase} phase")
    model.set_train()
    return model_structure(model)


def check_gradients(model, phase):
    active_has_gradient = False
    other = "vq" if phase == "deg" else "deg"
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if f"_{other}." in name:
            raise ValueError(f"Inactive LoRA branch received a gradient: {name}")
        if not torch.isfinite(gradient).all():
            raise ValueError(f"Nonfinite training gradient: {name}")
        if f"_{phase}." in name and torch.any(gradient != 0):
            active_has_gradient = True
    if not active_has_gradient:
        raise ValueError(f"Active {phase} LoRA branch has no nonzero gradient")
