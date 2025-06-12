import torch
from torch import Tensor
from torch_geometric.nn import global_mean_pool as scatter
from torch_geometric.utils import scatter
from typing import Optional


def global_var_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by computing the variance node features
    across the node dimension.

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2

    if batch is None:
        mean_value = x.mean(dim=dim, keepdim=x.dim() <= 2)
        mean_squared = (x * x).mean(dim=dim, keepdim=x.dim() <= 2)
    else:
        mean_value = scatter(x, batch, dim=dim, dim_size=size, reduce='mean')
        mean_squared = scatter(x * x, batch, dim=dim, dim_size=size, reduce='mean')
    var_value = (mean_squared - mean_value * mean_value).clamp(min=0)
    return var_value