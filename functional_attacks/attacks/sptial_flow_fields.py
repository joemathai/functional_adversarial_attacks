import torch


class SpatialFlowFields(torch.nn.Module):
    """
    An implementation of spatially transformed adversarial examples https://arxiv.org/pdf/1801.02612.pdf
    """
    def __init__(self, batch_shape, grid_scale_factor=1, step_size=0.1, pixel_shift_budget=2, random_init=False):
        """
        :param batch_shape:
        :param c:
        :param h:
        :param w:
        :param grid_scale_factor: resolution of the grid that is being perturbed (1 would mean every pixel is perturbed in
        isolation
        """
        super().__init__()
        batch_size, c, h, w = batch_shape
        assert h % grid_scale_factor == w % grid_scale_factor == 0, "height and width should be divisible by grid size"
        self.grid_size = grid_scale_factor
        self.step_size = step_size
        self.h_per_pixel_shift = (2.0 / h) * pixel_shift_budget  # grid generated is from [-1, 1]
        self.w_per_pixel_shift = (2.0 / w) * pixel_shift_budget
        self.register_buffer('identity_params',
                             torch.nn.functional.affine_grid(theta=torch.eye(2, 3).repeat(batch_size, 1, 1),
                                                             size=(batch_size, c, h // grid_scale_factor, w // grid_scale_factor),
                                                             align_corners=False),
                             persistent=False
                             )
        if random_init:
            min_shift = min(self.h_per_pixel_shift, self.w_per_pixel_shift)
            self.xform_params = torch.nn.Parameter(torch.empty(*self.identity_params.shape,
                                                               dtype=torch.float32).copy_(self.identity_params) +
                                                   torch.empty_like(self.identity_params, dtype=torch.float32).uniform_(
                                                       -min_shift, min_shift))
        else:
            self.xform_params = torch.nn.Parameter(torch.empty(*self.identity_params.shape,
                                                               dtype=torch.float32).copy_(self.identity_params))

    def forward(self, imgs):
        if self.grid_size != 1:
            grid = torch.nn.functional.interpolate(self.xform_params.permute(0, 3, 1, 2), scale_factor=self.grid_size,
                                                   mode='bicubic').permute(0, 2, 3, 1)
        else:
            grid = self.xform_params
        return torch.nn.functional.grid_sample(imgs, grid, align_corners=False)

    @torch.no_grad()
    def update_and_project_params(self):
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # standard linf measure is not applicable here
        # based on co-ordinate shift clip the parameters
        x_shift_clip_params = torch.unsqueeze(torch.clamp(
            self.xform_params[:, :, :, 0] - self.identity_params[:, :, :, 0], min=-self.w_per_pixel_shift,
            max=self.w_per_pixel_shift), dim=3)
        y_shift_clip_params = torch.unsqueeze(torch.clamp(
            self.xform_params[:, :, :, 1] - self.identity_params[:, :, :, 1], min=-self.h_per_pixel_shift,
            max=self.h_per_pixel_shift), dim=3)
        shift_clip_params = self.identity_params + torch.cat([x_shift_clip_params, y_shift_clip_params], dim=3)
        self.xform_params.copy_(shift_clip_params.clamp(min=-1.0, max=1.0))
