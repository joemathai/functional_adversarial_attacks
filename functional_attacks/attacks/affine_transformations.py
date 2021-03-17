import torch
import torchvision


class AffineTransforms(torch.nn.Module):
    """
    An affine transformation based attack
    """
    def __init__(self, batch_size, step_size=0.01):
        super().__init__()
        self.step_size = step_size
        self.xform_params = torch.nn.Parameter(torch.eye(2, 3).unsqueeze(dim=0).repeat(batch_size, 1, 1))

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

