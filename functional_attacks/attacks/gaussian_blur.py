import torch
from torch.nn.functional import conv2d, pad as torch_pad


class GaussianBlur(torch.nn.Module):
    def __init__(self, batch_shape, kernel_size=5, step_size=0.1, sigma_bounds=(0.2, 2.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.step_size = step_size
        self.sigma_bounds = sigma_bounds
        self.xform_params = torch.nn.Parameter(torch.ones(batch_shape[0], 2))

    @staticmethod
    def _get_gaussian_kernel1d(kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(sigma.device)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        return kernel1d

    @staticmethod
    def _get_gaussian_kernel2d(kernel_size, sigma, dtype, device):
        kernel1d_x = GaussianBlur._get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
        kernel1d_y = GaussianBlur._get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
        kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
        return kernel2d

    @staticmethod
    def _gaussian_blur(img, kernel_size, sigma, dtype=torch.float32):
        kernel = GaussianBlur._get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
        kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])
        # padding = (left, right, top, bottom)
        padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
        img = torch_pad(img.unsqueeze(0), padding, mode="reflect")
        img = conv2d(img, kernel, groups=img.shape[-3])
        return img

    def forward(self, imgs):
        blurred_imgs = list()
        for idx, img in enumerate(imgs):
            blurred_imgs.append(
                GaussianBlur._gaussian_blur(img, kernel_size=[self.kernel_size, self.kernel_size],
                                            sigma=self.xform_params[idx])
            )
        return torch.cat(blurred_imgs, dim=0)

    @torch.no_grad()
    def update_and_project_params(self):
        # update parameters
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # clamp the parameters to be within the bounds
        self.xform_params.copy_(torch.clamp(self.xform_params, min=self.sigma_bounds[0], max=self.sigma_bounds[1]))
