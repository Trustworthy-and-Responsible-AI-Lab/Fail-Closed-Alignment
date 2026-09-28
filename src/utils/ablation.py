import torch
import torch.nn as nn


def project_out_subspace(X, Q):
    """
    X: (..., d) activations (last dim is feature dim)
    Q: (d, r) orthonormal basis columns, or None
    Returns X with the span(Q) removed from the last dimension.
    """
    return X - (X @ Q) @ Q.transpose(0, 1)


class MultiFeatureAblation(nn.Module):
    def __init__(self, module, mode="all", dtype=torch.bfloat16) -> None:
        super().__init__()
        self.module = module
        self.dtype = dtype
        self.mode = mode
        self.Q = None  # Orthonormal basis for directions to project out

    @torch.no_grad()
    def update_directions(self, directions):
        """
        directions: 
          - Tensor (k, d) or (d, k), or list/tuple of k tensors each (d,)
        Updates the directions to project out.
        """
        if self.mode == "last":
            directions = [directions[-1]]
        D = torch.stack(directions, dim=1).to(dtype=torch.float32)  # (d, k)
        Q = torch.linalg.qr(D, mode='reduced').Q  # (d, r) with orthonormal columns
        self.Q = Q.to(device=next(self.module.parameters()).device, dtype=self.dtype)

    def ablate(self):
        """
        directions: 
          - Tensor (k, d) or (d, k), or list/tuple of k tensors each (d,)
        Projects all given directions out at once from inputs/outputs.
        """
        for layer in self.module.layers:
            self._ablate_input(layer, self.Q)
            self._ablate_output(layer.self_attn, self.Q, tuple_length=3)
            self._ablate_output(layer.mlp, self.Q, tuple_length=1)

    def _ablate_output(self, layer, Q, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0]
        else:
            activation = layer.output
        new_activation = project_out_subspace(activation, Q)
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        else:
            layer.output = new_activation

    def _ablate_input(self, layer, Q):
        new_activation = project_out_subspace(layer.input, Q)
        layer.input = new_activation