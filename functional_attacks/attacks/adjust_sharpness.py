import torch
import torchvision


class AdjustSharpness(torch.nn.Module):
    def __init__(self, batch_size, step_size=0.1, sharpness_bounds=(0.0, 5.0)):
        super().__init__()
        self.step_size = step_size
        self.sharpness_bounds = sharpness_bounds
        self.xform_params = torch.nn.Parameter(torch.ones(batch_size))

    def forward(self, imgs):
        sharpned_imgs = list()
        for idx, img in enumerate(imgs):
            img1 = imgs[idx]
            img2 = torchvision.transforms.functional_tensor._blurred_degenerate_image(imgs[idx])
            sharpned_imgs.append((self.xform_params[idx] * img1 +
                                  (1.0 - self.xform_params[idx]) * img2).clamp(0, 1).to(img1.dtype).unsqueeze_(dim=0))
        return torch.cat(sharpned_imgs, dim=0)

    @torch.no_grad()
    def update_and_project_params(self):
        # update parameters
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # clamp the sharpness parameters
        self.xform_params.copy_(torch.clamp(self.xform_params, min=self.sharpness_bounds[0], max=self.sharpness_bounds[1]))
