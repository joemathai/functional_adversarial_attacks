import torch


class AdjustGamma(torch.nn.Module):
    """
    A thin layer of convolutional kernels used to attack the image
    """
    def __init__(self, batch_shape, step_size=0.05, gamma_bounds=(0.3, 3.2), gain=1.0, random_init=False):
        super().__init__()
        self.step_size = step_size
        self.gamma_bounds = gamma_bounds
        self.gain = gain
        gamma = torch.ones(batch_shape[0], dtype=torch.float32).unsqueeze_(dim=1)
        if random_init:
            gamma += torch.empty_like(gamma).uniform_(*gamma_bounds)
        self.xform_params = torch.nn.Parameter(gamma)

    def forward(self, imgs):
        """
        formula used for adjusting gamma is x' = gain * x ^ gamma
        """
        b, c, h, w = imgs.shape
        return self.gain * torch.pow(imgs.reshape(b, -1), self.xform_params.view(-1, 1)).reshape(b, c, h, w)

    @torch.no_grad()
    def update_and_project_params(self):
        # update parameters
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # clamp gamma
        self.xform_params.copy_(torch.clamp(self.xform_params, min=self.gamma_bounds[0], max=self.gamma_bounds[1]))
