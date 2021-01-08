import torch


class SpatialFlowFields(torch.nn.Module):
    """
    An implementation of spatially transformed adversarial examples https://arxiv.org/pdf/1801.02612.pdf
    """
    def __init__(self, batch_size, c, h, w, grid_scale_factor=1):
        """
        :param batch_size:
        :param c:
        :param h:
        :param w:
        :param grid_scale_factor: resolution of the grid that is being perturbed (1 would mean every pixel is perturbed in
        isolation
        """
        super().__init__()
        assert h % grid_scale_factor == w % grid_scale_factor == 0, "height and width should be divisible by grid size"
        self.grid_size = grid_scale_factor
        self.register_buffer('identity_params',
                             torch.nn.functional.affine_grid(theta=torch.eye(2, 3).repeat(batch_size, 1, 1),
                                                             size=(batch_size, c, h // grid_scale_factor, w // grid_scale_factor),
                                                             align_corners=False),
                             persistent=False
                             )
        self.xform_params = torch.nn.Parameter(torch.empty(*self.identity_params.shape,
                                                           dtype=torch.float32).copy_(self.identity_params))

    def forward(self, imgs):
        if self.grid_size != 1:
            grid = torch.nn.functional.interpolate(self.xform_params.permute(0, 3, 1, 2), scale_factor=self.grid_size,
                                                   mode='bilinear').permute(0, 2, 3, 1)
        else:
            grid = self.xform_params
        return torch.nn.functional.grid_sample(imgs, grid, align_corners=False)
