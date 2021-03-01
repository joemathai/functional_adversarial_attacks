import torch
import torchvision
from PIL import Image
import numpy as np


class ConvolutionalKernel(torch.nn.Module):
    """
    A thin layer of convolutional kernels used to attack the image
    """
    def __init__(self, batch_size, input_channels=3, output_channels=3):
        super().__init__()
        self.batch_size = batch_size
        weights = torch.empty(output_channels, input_channels, 3, 3)
        # fill the kernel with identity preserving function
        torch.nn.init.dirac_(weights)
        # # for each image create a convolutional weight tensor (the weights are not shared)
        weights = weights.repeat(batch_size, 1, 1, 1, 1)
        self.xform_params = torch.nn.Parameter(weights)
        self.iter = 0
    
    @staticmethod
    def _viz_filters(weights, idx, iter):
        grid_img = (torchvision.utils.make_grid(
            torch.nn.functional.upsample(weights.clone().detach().cpu(), scale_factor=40), padding=3, normalize=False,
            pad_value=1).numpy() * 255).transpose((1, 2, 0))
        Image.fromarray(np.uint8(grid_img)).save(f'cnn_filters/{idx}_{iter}.png')

    def forward(self, imgs):
        self.iter += 1
        convolved_imgs = []
        for i in range(imgs.shape[0]):
            convolved_imgs.append(torch.nn.functional.conv2d(torch.unsqueeze(imgs[i], dim=0),
                                                             self.xform_params[i].view(3, 3, 3, 3), bias=None,
                                                             stride=1, padding=1))

            # # visualization of the filters and the images associated with the filters
            # if self.iter == 20:
            #     with torch.no_grad():
            #         self._viz_filters(self.xform_params[i], i, self.iter)
            #         c, h, w = imgs[i].shape
            #         r, g, b = torch.unbind(convolved_imgs[-1].squeeze().detach().cpu(), dim=0)
            #         r_img = torch.cat([r.view(1, h, w), torch.zeros(2, h, w)], dim=0).unsqueeze(dim=0)
            #         g_img = torch.cat([torch.zeros(1, h, w), g.view(1, h, w), torch.zeros(1, h, w)], dim=0).unsqueeze(dim=0)
            #         b_img = torch.cat([torch.zeros(1, h, w), torch.zeros(1, h, w), b.view(1, h, w)], dim=0).unsqueeze(dim=0)
            #         print(r_img.shape, g_img.shape, b_img.shape)
            #         grid_rgb = (torchvision.utils.make_grid(torch.abs(torch.cat([r_img, g_img, b_img], dim=0)), padding=3, normalize=False).numpy() * 255).transpose((1, 2, 0))
            #         Image.fromarray(np.uint8(grid_rgb)).save(f'cnn_filtered_imgs/{i}.png')

        return torch.abs(torch.cat(convolved_imgs, dim=0))


# class ConvolutionalKernel(torch.nn.Module):
#     def __init__(self, batch_size):
#         super().__init__()
#         self.batch_size = batch_size
#         weights = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
#         weights = weights.view(1, 1, 1, 3, 3).repeat(batch_size, 3, 1, 1, 1)
#
#         self.xform_params = torch.nn.Parameter(weights)
#         self.iter = 0
#
#     def _viz_filters(self, weights, idx):
#         import torchvision
#         from PIL import Image, ImageDraw, ImageFont
#         import numpy as np
#         grid_img = (torchvision.utils.make_grid(
#             torch.nn.functional.upsample(weights.clone().detach().cpu(), scale_factor=40), padding=3, normalize=True,
#             pad_value=1).numpy() * 255).transpose((1, 2, 0))
#         # Image.fromarray(np.uint8(grid_img)).save(f'cnn_filters/{idx}_{self.iter}.png')
#         image = Image.fromarray(np.uint8(grid_img))
#         draw = ImageDraw.Draw(image)
#         draw.text((0, 0), f"{list(np.around(weights[0].view(-1).detach().cpu().numpy(), decimals=2))}", (0, 255, 0))
#         draw.text((0, 30), f"{list(np.around(weights[1].view(-1).detach().cpu().numpy(), decimals=2))}", (0, 255, 0))
#         draw.text((0, 60), f"{list(np.around(weights[2].view(-1).detach().cpu().numpy(), decimals=2))}", (0, 255, 0))
#         # draw.text((0, 0), f"{weights[1].view(-1)}", (255, 255, 255))
#         # draw.text((0, 0), f"{weights[1].view(-1)}", (255, 255, 255))
#         image.save(f'cnn_filters/img_no_{idx}_iter_no_{self.iter}.png')
#
#     def forward(self, imgs):
#         r, g, b = torch.unbind(imgs, dim=1)
#         # convert to grayscale
#         imgs = (0.2989 * r + 0.587 * g + 0.114 * b).to(imgs.dtype).unsqueeze(dim=1)
#         convolved_imgs = list()
#         self.iter += 1
#         for i in range(imgs.shape[0]):
#             self._viz_filters(self.xform_params[i], i)
#             convolved_imgs.append(torch.nn.functional.conv2d(torch.unsqueeze(imgs[i], dim=0),
#                                                              self.xform_params[i].view(3, 1, 3, 3), bias=None,
#                                                              stride=1, padding=1))
#         combined_img = torch.cat(convolved_imgs, dim=0)
#         return torch.abs(combined_img)
