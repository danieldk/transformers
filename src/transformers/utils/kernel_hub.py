from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Architecture:
    device: str

    def __eq__(self, other):
        return isinstance(other, Architecture) and self.device == other.device

    def __hash__(self):
        return hash(self.device)


@dataclass
class LayerRepository:
    layer: str
    repo_id: str
    revision: str


KERNEL_MAPPING: Dict[str, Dict[Architecture, LayerRepository]] = {}


def use_hub_kernel(kernel_name: str):
    def decorator(cls):
        # Lookup kernels.
        kernel = KERNEL_MAPPING.get(kernel_name)
        if kernel is None:
            return cls

        # Where can we get the device other than from the input?
        device = "cuda"
        repo = kernel.get(Architecture(device=device))
        if repo is None:
            return cls

        if repo is not None:
            try:
                print(f"Using layer repo {repo}")
                return _get_kernel_layer(
                    repo_id=repo.repo_id,
                    layer_name=repo.layer_name,
                    revision=repo.revision,
                )
            except Exception as _:
                return cls

        # No kernel available, return original implemenation.
        print(f"No kernel available for {kernel_name}, falling back")
        return cls

    return decorator


def _get_kernel_layer(*, repo_id: str, layer_name: str, revision: str) -> "nn.Module":
    """Get a layer from a kernel."""

    from kernels import get_kernel
    from torch import nn

    kernel = get_kernel(repo_id, revision=revision)

    if getattr(kernel, "layers", None) is None:
        raise ValueError(
            f"Kernel `{repo_id}` at revision `{revision}` does not define any layers."
        )

    layer = getattr(kernel.layers, layer_name, None)
    if layer is None:
        raise ValueError(f"Layer `{layer_name}` not found in kernel `{repo_id}`.")
    if not issubclass(layer, nn.Module):
        raise TypeError(f"Layer `{layer_name}` is not a Torch layer.")
    return layer
