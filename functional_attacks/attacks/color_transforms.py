import torch


class SWIRColorTransforms(torch.nn.Module):
    """
    Channel invariant non-linear color transform for SWIR or grayscale images
    """
    def __init__(self, batch_shape, resolution=32, step_size=0.003, linf_budget=0.03, random_init=False):
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
    RGB -> CIELUV color space is not yet implemented
    """

    def __init__(self, batch_shape, resolution_x=32, resolution_y=32, resolution_z=32,
                 step_size=0.003, linf_budget=0.03, random_init=False):
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
        return result.permute(0, 3, 1, 2)

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

