import torch


class AdjustBrightnessContrast(torch.nn.Module):
    """
    A thin layer of convolutional kernels used to attack the image
    """
    def __init__(self, batch_shape, brightness_step_size=3/255., contrast_step_size=0.03,
                 brightness_bounds=(-0.5, 0.5), contrast_bounds=(0.5, 1.5)):
        super().__init__()
        self.batch_shape = batch_shape
        self.brightness_step_size = brightness_step_size
        self.contrast_step_size = contrast_step_size
        self.brightness_bounds = brightness_bounds
        self.contrast_bounds = contrast_bounds
        alpha = torch.ones(self.batch_shape[0], dtype=torch.float32).unsqueeze_(dim=1)
        beta = torch.zeros(self.batch_shape[0], dtype=torch.float32).unsqueeze_(dim=1)
        self.xform_params = torch.nn.Parameter(torch.cat([alpha, beta], dim=1))

    def forward(self, imgs):
        """
        formula used for adjusting contrast and brightness is x' = alpha(x - 0.5) + 0.5 + beta
        """
        b, c, h, w = imgs.shape
        return torch.clamp(self.xform_params[:, 0].view(-1, 1) * (imgs.reshape(b, -1) - 0.5) + 0.5 +
                           self.xform_params[:, 1].view(-1, 1), min=0.0, max=1.0).reshape(b, c, h, w)

    @torch.no_grad()
    def update_and_project_params(self):
        # update parameters
        self.xform_params[:, 0].sub_(torch.sign(self.xform_params.grad[:, 0]) * self.contrast_step_size)
        self.xform_params[:, 1].sub_(torch.sign(self.xform_params.grad[:, 1]) * self.brightness_step_size)
        # make sure that the parameters are within the valid image bounds
        # clamp alpha (contrast param)
        self.xform_params[:, 0].copy_(torch.clamp(self.xform_params[:, 0],
                                                  min=self.contrast_bounds[0], max=self.contrast_bounds[1]))
        # clamp beta (brightness param)
        self.xform_params[:, 1].copy_(torch.clamp(self.xform_params[:, 1],
                                                  min=self.brightness_bounds[0], max=self.brightness_bounds[1]))
