import torch


class AffineTransforms(torch.nn.Module):
    """
    An affine transformation based attack
    """
    def __init__(self, batch_size):
        super().__init__()
        self.register_buffer('identity_params', torch.eye(2, 3).unsqueeze(dim=0).repeat(batch_size, 1, 1), persistent=False)
        self.xform_params = torch.nn.Parameter(torch.eye(2, 3).unsqueeze(dim=0).repeat(batch_size, 1, 1))

    def forward(self, imgs):
        affine_grid = torch.nn.functional.affine_grid(theta=self.xform_params, size=imgs.shape, align_corners=False)
        return torch.nn.functional.grid_sample(input=imgs, grid=affine_grid, mode='bilinear', align_corners=False)
