import torch


class Delta(torch.nn.Module):
    """
    A basic additive layer that transforms x = x + delta
    """
    def __init__(self, batch_shape, step_size=1/255, linf_budget=4/255, random_init=False):
        super().__init__()
        batch_size, c, h, w = batch_shape
        self.step_size = step_size
        self.linf_budget = linf_budget
        self.register_buffer('identity_params',
                             torch.zeros(batch_size, c, h, w, dtype=torch.float32),
                             persistent=False)
        if random_init:
            self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).uniform_(-linf_budget, linf_budget))
        else:
            self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).copy_(self.identity_params))

    def forward(self, imgs):
        return torch.clamp(imgs + self.xform_params, min=0.0, max=1.0)

    @torch.no_grad()
    def update_and_project_params(self):
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # based on linf budget clip the parameters
        self.xform_params.copy_(
            (self.xform_params - self.identity_params).clamp(min=-self.linf_budget, max=self.linf_budget) +
            self.identity_params
        )
