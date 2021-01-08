import torch


def l2_grid_smoothness(grid):
    """
    Given a variable of dimensions (N, X, Y, [Z], C), computes the sum of
    the differences between adjacent points in the grid formed by the
    dimensions X, Y, and (optionally) Z. Returns a tensor of dimension N.
    reference: https://arxiv.org/pdf/1801.02612.pdf (4)
    """
    num_dims = len(grid.size()) - 2
    batch_size = grid.size()[0]
    norm = torch.zeros(batch_size, dtype=grid.data.dtype, device=grid.data.device, requires_grad=True)
    for dim in range(num_dims):
        slice_before = (slice(None),) * (dim + 1)
        slice_after = (slice(None),) * (num_dims - dim)
        shifted_grids = [
            # left
            torch.cat([
                grid[slice_before + (slice(1, None),) + slice_after],
                grid[slice_before + (slice(-1, None),) + slice_after],
            ], dim + 1),
            # right
            torch.cat([
                grid[slice_before + (slice(None, 1),) + slice_after],
                grid[slice_before + (slice(None, -1),) + slice_after],
            ], dim + 1)
        ]
        for shifted_grid in shifted_grids:
            norm_componenets = torch.norm(shifted_grid - grid, p=2, dim=len(shifted_grid.shape) - 1)
            norm = norm + norm_componenets.sum(tuple(range(1, len(norm_componenets.shape))))
    return norm