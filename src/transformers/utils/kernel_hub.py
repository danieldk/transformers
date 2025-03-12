from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Architecture:
    device_type: str

    def __eq__(self, other):
        return isinstance(other, Architecture) and self.device_type == other.device_type

    def __hash__(self):
        return hash(self.device_type)


@dataclass
class LayerRepository:
    layer_name: str
    repo_id: str
    revision: str = "main"


KERNEL_MAPPING: Dict[str, Dict[Architecture, LayerRepository]] = {}


def use_hub_kernel(kernel_name: str):
    def decorator(cls):
        # Lookup kernels.
        kernel = KERNEL_MAPPING.get(kernel_name)
        if kernel is None:
            return cls

        # Where can we get the device other than from the input?
        device = "cuda"
        repo = kernel.get(Architecture(device_type=device))
        if repo is None:
            return cls

        try:
            print(f"Using layer repo {repo}")
            return _get_kernel_layer(
                repo_id=repo.repo_id,
                layer_name=repo.layer_name,
                revision=repo.revision,
            )
        except Exception as _:
            return cls

    return decorator


def use_hub_kernel_forward(kernel_name: str):
    def decorator(cls):
        fallback_forward = cls.forward

        cached_forward = {}

        def forward(self, x, **args):
            kernel = KERNEL_MAPPING.get(kernel_name)
            if kernel is None:
                print("No kernel mapping")
                return fallback_forward(self, x, **args)

            device = getattr(x, "device", None)
            if device is None:
                print("No kernel for device")
                return fallback_forward(self, x, **args)

            print(kernel)
            arch = Architecture(device_type=device.type)

            print(arch, cached_forward)

            # Short-circuit if we already loaded the layer.
            layer_forward = cached_forward.get(arch, None)
            if layer_forward is not None:
                print("hit cache")
                return layer_forward(self, x, **args)

            repo = kernel.get(arch)
            if repo is None:
                print("No repo?")
                return fallback_forward(self, x, **args)

            try:
                print(f"Using layer repo {repo}")
                layer = _get_kernel_layer(
                    repo_id=repo.repo_id,
                    layer_name=repo.layer_name,
                    revision=repo.revision,
                )
                layer_forward = layer.forward
                cached_forward[arch] = layer_forward
            except Exception as e:
                print(e)
                layer_forward = fallback_forward

            return layer_forward(self, x, **args)

        print(f"overriding forward: {cls.forward}")
        cls.forward = forward
        print(f"after forward: {cls.forward}")

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
