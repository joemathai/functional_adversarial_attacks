import torch
import numpy as np


class IndependentChannelColorTransforms(torch.nn.Module):
    """
    Channel invariant non-linear color transform for SWIR or grayscale images
    """
    def __init__(self, batch_shape, resolution=72, step_size=1/255, linf_budget=10/255, random_init=False):
        super().__init__()
        self.resolution = resolution
        self.step_size = step_size
        self.linf_budget = linf_budget
        self.register_buffer('identity_params', torch.empty(batch_shape[0], resolution, 3, dtype=torch.float32),
                             persistent=False)
        for x in range(resolution):
            self.identity_params[:, x, :] = x / (resolution - 1)

        if random_init:
            random_delta = torch.empty(batch_shape[0], resolution, 3).uniform_(-self.linf_budget, self.linf_budget)
            self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).copy_(self.identity_params) + random_delta)
        else:
            self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).copy_(self.identity_params))

    def forward(self, imgs):
        N, C, H, W = imgs.shape
        imgs = imgs.permute(0, 2, 3, 1) # N, H, W, C
        imgs_scaled = imgs * torch.tensor([self.resolution - 1] * C, dtype=torch.float32, device=self.xform_params.device)[None, None, None, :].expand(N, H, W, C)
        integer_part, float_part = torch.floor(imgs_scaled).long(), imgs_scaled % 1
        transformed_channels = list()
        for ch in range(3):
            endpoints = list()
            for delta_x in [0, 1]:
                color_index = torch.clamp(integer_part[:, :, :, ch] + delta_x, 0, self.resolution - 1).reshape(N, -1)
                param_index = torch.gather(self.xform_params[:, :, ch], 1, color_index).view(N, H, W, 1)
                endpoints.append(param_index)
            transformed_channels.append(
                torch.clamp(endpoints[0] * (1 - float_part[:, :, :, ch].unsqueeze(-1)) +
                            endpoints[1] * float_part[:, :, :, ch].unsqueeze(-1), 0, 1).permute(0, 3, 1, 2)
            )
        return torch.clamp(torch.cat(transformed_channels, dim=1), 0, 1.0)

    @torch.no_grad()
    def update_and_project_params(self):
        # update params
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # clip the parameters to be within pixel intensities [0, 1]
        self.xform_params.copy_(torch.clamp(self.xform_params, min=0.0, max=1.0))
        # based on linf budget clip the parameters and project
        self.xform_params.copy_(
            (self.xform_params - self.identity_params).clamp(min=-self.linf_budget, max=self.linf_budget) +
            self.identity_params
        )


class SWIRColorTransforms(torch.nn.Module):
    """
    Channel invariant non-linear color transform for SWIR or grayscale images
    """
    def __init__(self, batch_shape, resolution=64, step_size=0.003, linf_budget=0.1, random_init=False):
        super().__init__()
        self.resolution = resolution
        self.step_size = step_size
        self.linf_budget = linf_budget
        self.register_buffer('identity_params', torch.empty(batch_shape[0], resolution, dtype=torch.float32),
                             persistent=False)
        for x in range(resolution):
            self.identity_params[:, x] = x / (resolution - 1)

        if random_init:
            random_delta = torch.empty(batch_shape[0], resolution).uniform_(-self.linf_budget, self.linf_budget)
            self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).copy_(self.identity_params) + random_delta)
        else:
            self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).copy_(self.identity_params))

    def forward(self, imgs):
        N, C, H, W = imgs.shape
        imgs = imgs.permute(0, 2, 3, 1) # N, H, W, C
        imgs_scaled = imgs * torch.tensor([self.resolution - 1] * C, dtype=torch.float32, device=self.xform_params.device)[None, None, None, :].expand(N, H, W, C)
        integer_part, float_part = torch.floor(imgs_scaled).long(), imgs_scaled % 1
        endpoints = list()
        for delta_x in [0, 1]:
            color_index = torch.clamp(integer_part + delta_x, 0, self.resolution - 1).reshape(N, -1)
            param_index = torch.gather(self.xform_params, 1, color_index).view(N, H, W, C)
            endpoints.append(param_index)
        return torch.clamp(endpoints[0] * (1 - float_part) + endpoints[1] * float_part, 0, 1).permute(0, 3, 1, 2)

    @torch.no_grad()
    def update_and_project_params(self):
        # update params
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # clip the parameters to be within pixel intensities [0, 1]
        self.xform_params.copy_(torch.clamp(self.xform_params, min=0.0, max=1.0))
        # based on linf budget clip the parameters and project
        self.xform_params.copy_(
            (self.xform_params - self.identity_params).clamp(min=-self.linf_budget, max=self.linf_budget) +
            self.identity_params
        )


class ColorTransforms(torch.nn.Module):
    """
    An implementation of functional adversarial attack https://papers.nips.cc/paper/2019/file/6e923226e43cd6fac7cfe1e13ad000ac-Paper.pdf
    The code is taken from https://github.com/cassidylaidlaw/ReColorAdv
    """

    def __init__(self, batch_shape, resolution_x=72, resolution_y=72, resolution_z=72,
                 step_size=1/255, linf_budget=10/255, random_init=False):
        super().__init__()
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.step_size = step_size
        self.linf_budget = linf_budget
        # construct identity parameters
        self.register_buffer('identity_params',
                             torch.empty(batch_shape[0], resolution_x, resolution_y, resolution_z, 3, dtype=torch.float32),
                             persistent=False)
        for x in range(resolution_x):
            for y in range(resolution_y):
                for z in range(resolution_z):
                    self.identity_params[:, x, y, z, 0] = x / (resolution_x - 1)
                    self.identity_params[:, x, y, z, 1] = y / (resolution_y - 1)
                    self.identity_params[:, x, y, z, 2] = z / (resolution_z - 1)

        # if random_init then add uniform noise to the xform param within the linf_budget
        if random_init:
            random_delta = torch.empty(batch_shape[0], resolution_x, resolution_y, resolution_z, 1).uniform_(-self.linf_budget, self.linf_budget)
            self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).copy_(self.identity_params) + random_delta)
        else:
            self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).copy_(self.identity_params))

    def forward(self, imgs):
        N, C, H, W = imgs.shape
        imgs = imgs.permute(0, 2, 3, 1)  # N x H x W x C
        # multiply each r, g, b with resolution_x, resolution_y, resolution_z
        imgs_scaled = imgs * torch.tensor([self.resolution_x - 1, self.resolution_y - 1, self.resolution_z - 1],
                                          dtype=torch.float32,
                                          device=self.xform_params.device)[None, None, None, :].expand(N, H, W, C)
        integer_part, float_part = torch.floor(imgs_scaled).long(), imgs_scaled % 1
        params_list = self.xform_params.view(N, -1, 3)

        # trilinear interpolation
        endpoint_values = []
        for delta_x in [0, 1]:
            corner_values = []

            for delta_y in [0, 1]:
                vertex_values = []

                for delta_z in [0, 1]:
                    params_index = torch.autograd.Variable(
                        torch.zeros(N, H, W, dtype=torch.long, device=self.xform_params.device)
                    )

                    for color_index, resolution in [
                        (integer_part[..., 0] + delta_x, self.resolution_x),
                        (integer_part[..., 1] + delta_y, self.resolution_y),
                        (integer_part[..., 2] + delta_z, self.resolution_z)
                    ]:
                        color_index = color_index.clamp(0, resolution - 1)
                        params_index = (params_index * resolution + color_index)

                    params_index = params_index.view(N, -1)[:, :, None].expand(-1, -1, 3)
                    vertex_values.append(params_list.gather(1, params_index).view(N, H, W, C))

                corner_values.append(
                    vertex_values[0] * (1 - float_part[..., 2, None]) +
                    vertex_values[1] * float_part[..., 2, None])

            endpoint_values.append(
                corner_values[0] * (1 - float_part[..., 1, None]) +
                corner_values[1] * float_part[..., 1, None]
            )

        result = endpoint_values[0] * (1 - float_part[..., 0, None]) + endpoint_values[1] * float_part[..., 0, None]
        return torch.clamp(result.permute(0, 3, 1, 2), min=0.0, max=1.0)

    @torch.no_grad()
    def update_and_project_params(self):
        # update params
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # clip the parameters to be within pixel intensities [0, 1]
        self.xform_params.copy_(torch.clamp(self.xform_params, min=0.0, max=1.0))
        # based on linf budget clip the parameters and project
        self.xform_params.copy_(
            (self.xform_params - self.identity_params).clamp(min=-self.linf_budget, max=self.linf_budget) +
            self.identity_params
        )


class ColorTransformsCIELUV(torch.nn.Module):

    class CIEXYZColorSpace:
        """
        The 1931 CIE XYZ color space (assuming input is in sRGB).
        
        Warning: may have values outside [0, 1] range. Should only be used in
        the process of converting to/from other color spaces.
        """

        def from_rgb(self, imgs):
            # apply gamma correction
            small_values_mask = (imgs < 0.04045).float()
            imgs_corrected = (
                (imgs / 12.92) * small_values_mask +
                ((imgs + 0.055) / 1.055) ** 2.4 * (1 - small_values_mask)
            )
            # linear transformation to XYZ
            r, g, b = imgs_corrected.permute(1, 0, 2, 3)
            x = 0.4124 * r + 0.3576 * g + 0.1805 * b
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
            z = 0.0193 * r + 0.1192 * g + 0.9504 * b
            return torch.stack([x, y, z], 1)

        def to_rgb(self, imgs):
            # linear transformation
            x, y, z = imgs.permute(1, 0, 2, 3)
            r = 3.2406 * x - 1.5372 * y - 0.4986 * z
            g = -0.9689 * x + 1.8758 * y + 0.0415 * z
            b = 0.0557 * x - 0.2040 * y + 1.0570 * z
            imgs = torch.stack([r, g, b], 1)
            # apply gamma correction
            small_values_mask = (imgs < 0.0031308).float()
            imgs_clamped = imgs.clamp(min=1e-10)  # prevent NaN gradients
            imgs_corrected = (
                (12.92 * imgs) * small_values_mask +
                (1.055 * imgs_clamped ** (1 / 2.4) - 0.055) *
                (1 - small_values_mask)
            )
            return imgs_corrected


    class CIELUVColorSpace:
        """
        Converts to the 1976 CIE L*u*v* color space.
        """
        def __init__(self, up_white=0.1978, vp_white=0.4683, y_white=1,
                     eps=1e-10):
            self.xyz_cspace = ColorTransformsCIELUV.CIEXYZColorSpace()
            self.up_white = up_white
            self.vp_white = vp_white
            self.y_white = y_white
            self.eps = eps

        def from_rgb(self, imgs):
            x, y, z = self.xyz_cspace.from_rgb(imgs).permute(1, 0, 2, 3)
            # calculate u' and v'
            denom = x + 15 * y + 3 * z + self.eps
            up = 4 * x / denom
            vp = 9 * y / denom
            # calculate L*, u*, and v*
            small_values_mask = (y / self.y_white < (6 / 29) ** 3).float()
            y_clamped = y.clamp(min=self.eps)  # prevent NaN gradients
            L = (
                ((29 / 3) ** 3 * y / self.y_white) * small_values_mask +
                (116 * (y_clamped / self.y_white) ** (1 / 3) - 16) *
                (1 - small_values_mask)
            )
            u = 13 * L * (up - self.up_white)
            v = 13 * L * (vp - self.vp_white)
            return torch.stack([L / 100, (u + 100) / 200, (v + 100) / 200], 1)

        def to_rgb(self, imgs):
            L = imgs[:, 0, :, :] * 100
            u = imgs[:, 1, :, :] * 200 - 100
            v = imgs[:, 2, :, :] * 200 - 100
            up = u / (13 * L + self.eps) + self.up_white
            vp = v / (13 * L + self.eps) + self.vp_white
            small_values_mask = (L <= 8).float()
            y = (
                (self.y_white * L * (3 / 29) ** 3) * small_values_mask +
                (self.y_white * ((L + 16) / 116) ** 3) * (1 - small_values_mask)
            )
            denom = 4 * vp + self.eps
            x = y * 9 * up / denom
            z = y * (12 - 3 * up - 20 * vp) / denom
            return self.xyz_cspace.to_rgb(
                torch.stack([x, y, z], 1).clamp(0, 1.1)).clamp(0, 1)
        
    def __init__(self, batch_shape, resolution_x=64, resolution_y=32, resolution_z=32,
                 step_size=1/255, linf_budget=10/255, random_init=False):
        super().__init__()
        self.luv_color_space = self.CIELUVColorSpace()
        self.adversary = ColorTransforms(batch_shape, resolution_x, resolution_y, resolution_z, random_init=random_init)

    def forward(self, imgs):
        return torch.clamp(self.luv_color_space.to_rgb(self.adversary(self.luv_color_space.from_rgb(imgs))), 0.0, 1.0)


class ColorTransformsHSV(torch.nn.Module):
    
    class ApproxHSVColorSpace:
        """
        Converts from RGB to approximately the HSV cone using a much smoother
        transformation.
        """
        def from_rgb(self, imgs):
            r, g, b = imgs.permute(1, 0, 2, 3)
            x = r * np.sqrt(2) / 3 - g / (np.sqrt(2) * 3) - b / (np.sqrt(2) * 3)
            y = g / np.sqrt(6) - b / np.sqrt(6)
            z, _ = imgs.max(1)
            return torch.stack([z, x + 0.5, y + 0.5], 1)

        def to_rgb(self, imgs):
            z, xp, yp = imgs.permute(1, 0, 2, 3)
            x, y = xp - 0.5, yp - 0.5
            rp = float(np.sqrt(2)) * x
            gp = -x / np.sqrt(2) + y * np.sqrt(3 / 2)
            bp = -x / np.sqrt(2) - y * np.sqrt(3 / 2)
            delta = z - torch.max(torch.stack([rp, gp, bp], 1), 1)[0]
            r, g, b = rp + delta, gp + delta, bp + delta
            return torch.stack([r, g, b], 1).clamp(0, 1)
    
    def __init__(self, batch_shape, resolution=72, step_size=1/255, linf_budget=6/255, random_init=False):
        super().__init__()
        self.hsv_color_space = self.ApproxHSVColorSpace()
        self.adversary = IndependentChannelColorTransforms(batch_shape, resolution, step_size=step_size,
                                                           linf_budget=linf_budget, random_init=random_init)

    def forward(self, imgs):
        return torch.clamp(self.hsv_color_space.to_rgb(self.adversary(self.hsv_color_space.from_rgb(imgs))), 0.0, 1.0)
        
        
