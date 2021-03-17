import math
import torch

# refer implementation in torchvision library for more details


def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values
    eqc = maxc == minc
    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img):
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)
    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6
    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)
    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


class AdjustHueSaturation(torch.nn.Module):
    def __init__(self, batch_size, step_size=0.1, saturation_bounds=(0.2, 2.0), hue_bounds=(-math.pi, math.pi)):
        super().__init__()
        self.step_size = step_size
        self.saturation_bounds = saturation_bounds
        self.hue_bounds = hue_bounds
        saturation_params = torch.ones(batch_size, dtype=torch.float32).unsqueeze_(dim=1)
        hue_params = torch.zeros(batch_size, dtype=torch.float32).unsqueeze_(dim=1)
        self.xform_params = torch.nn.Parameter(torch.cat([hue_params, saturation_params], dim=1))

    def forward(self, imgs):
        b, c, h, w = imgs.shape
        # unpack the hsv values
        hue, saturation, value = torch.chunk(_rgb2hsv(imgs), chunks=3, dim=-3)
        # adjust the hue
        mod_hue = torch.fmod(hue.view(b, -1) + self.xform_params[:, 0].view(-1, 1), 2 * math.pi).view(b, 1, h, w)
        # adjust the saturation
        mod_saturation = torch.clamp(saturation.view(b, -1) * self.xform_params[:, 1].view(-1, 1),
                                     min=0, max=1).view(b, 1, h, w)
        # pack back back the corrected hue
        return _hsv2rgb(torch.cat([mod_hue, mod_saturation, value], dim=-3))

    @torch.no_grad()
    def update_and_project_params(self):
        # update parameters
        self.xform_params.sub_(torch.sign(self.xform_params.grad) * self.step_size)
        # clamp hue and saturation params
        self.xform_params[:, 0].copy_(torch.clamp(self.xform_params[:, 0], min=self.hue_bounds[0],
                                                  max=self.hue_bounds[1]))
        self.xform_params[:, 1].copy_(torch.clamp(self.xform_params[:, 1], min=self.saturation_bounds[0],
                                                  max=self.saturation_bounds[1]))

