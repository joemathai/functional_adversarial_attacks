import torch


class Delta(torch.nn.Module):
    """
    A basic additive layer that transforms x = x + delta
    """
    def __init__(self, batch_size, c, h, w):
        """
        :param batch_size:
        :param c:
        :param h:
        :param w:
        """
        super().__init__()
        self.register_buffer('identity_params',
                             torch.zeros(batch_size, c, h, w, dtype=torch.float32),
                             persistent=False)
        self.xform_params = torch.nn.Parameter(torch.empty_like(self.identity_params).copy_(self.identity_params))

    def forward(self, imgs):
        return imgs + self.xform_params
