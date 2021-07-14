import torch
import numpy as np
import torchvision


class AffineTransforms(torch.nn.Module):
    """
    An affine transformation based attack
    """
    def __init__(self, batch_shape, step_size=0.01, random_init=False):
        super().__init__()
        self.step_size = step_size
        self.xform_params = torch.nn.Parameter(torch.eye(2, 3).unsqueeze(dim=0).repeat(batch_shape[0], 1, 1))
        if random_init:
            import warnings
            warnings.warn("using random_init=True is not supported for AffineTransforms")

    def forward(self, imgs):
        b, c, h, w = imgs.shape
        transformed_imgs = list()
        for idx, img in enumerate(imgs):
            theta = self.xform_params[idx].view(1, 2, 3)
            grid = torchvision.transforms.functional_tensor._gen_affine_grid(theta, w=w, h=h, ow=w, oh=h)
            transformed_imgs.append(
                torchvision.transforms.functional_tensor._apply_grid_transform(img, grid, 'bilinear',
                                                                               fill=None).unsqueeze_(dim=0))
        return torch.cat(transformed_imgs, dim=0)

    @torch.no_grad()
    def update_and_project_params(self):
        # update parameters
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)


class RotationTranslationTransforms(torch.nn.Module):
    """
    An affine transformation based attack
    """

    def __init__(self, batch_shape, angle_step_size=np.pi / 18, shift_step_size=0.02,
                 rot_bounds=(-np.pi / 2.5, np.pi / 2.5), shift_bounds=(-0.3, 0.3), random_init=False):
        super().__init__()
        self.angle_step_size = angle_step_size
        self.shift_step_size = shift_step_size
        if random_init:
            angles = torch.zeros(batch_shape[0], 1).uniform_(*rot_bounds)
            translation = torch.zeros(batch_shape[0], 2).uniform_(*shift_bounds)
            self.xform_params = torch.nn.Parameter(torch.cat([angles, translation], dim=1))
        else:
            self.xform_params = torch.nn.Parameter(torch.zeros(batch_shape[0], 3))

    def forward(self, imgs):
        b, c, h, w = imgs.shape
        sin_params = torch.sin(self.xform_params[:, 0]).view(-1, 1)
        cos_params = torch.cos(self.xform_params[:, 0]).view(-1, 1)
        tx, ty = self.xform_params[:, 1].view(-1, 1), self.xform_params[:, 2].view(-1, 1)
        affine_mats = torch.cat([cos_params, -sin_params, tx,
                                 sin_params, cos_params, ty],
                                dim=1).reshape(b, 2, 3)
        grid = torch.nn.functional.affine_grid(affine_mats, imgs.shape).to(imgs.device)
        return torch.nn.functional.grid_sample(imgs, grid)

    @torch.no_grad()
    def update_and_project_params(self):
        # update parameters
        self.xform_params[:, 0].sub_(torch.sign(self.xform_params.grad[:, 0]) * self.angle_step_size)
        self.xform_params[:, 1:].sub_(torch.sign(self.xform_params.grad[:, 1:]) * self.shift_step_size)
