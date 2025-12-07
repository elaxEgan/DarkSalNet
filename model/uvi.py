import torch
import torch.nn as nn
import torch.nn.functional as F


pi = 3.141592653589793


class RGB_UVI(nn.Module):
    def __init__(self):
        super(RGB_UVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))  # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0
        self.alpha_H = torch.nn.Parameter(torch.full([1], 0.1))
        self.alpha_V = torch.nn.Parameter(torch.full([1], 0.1))
        self.alpha_S = torch.nn.Parameter(torch.full([1], 0.1))
        self.contrast_conv = nn.Linear(1, 1, bias=False)

    def UVIT(self, img, h_lab, s_lab, a, b):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype

        # Extract HVI components: hue, saturation, value
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)

        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6

        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)

        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        chl = (2.0 * pi * h_lab).cos()
        cvl = (2.0 * pi * h_lab).sin()

        ch = (1 - self.alpha_H) * ch + self.alpha_H * chl.unsqueeze(1)
        cv = (1 - self.alpha_H) * cv + self.alpha_V * cvl.unsqueeze(1)
        saturation = (1 - self.alpha_S) * saturation + self.alpha_S * s_lab.unsqueeze(1)
        c = torch.sigmoid(self.color_contrast(a, b, 5).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        H = color_sensitive * saturation * ch * c
        V = color_sensitive * saturation * cv * c
        I = value
        xyz = torch.cat([H, V, I], dim=1)

        return xyz

    def PUVIT(self, img):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # Clip values
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha

        return rgb

    def lab_to_uvi(self, lab_img):
        # Extract a* and b* from Lab
        lab_img = lab_img.permute(0, 3, 1, 2)
        a_star = lab_img[:, 1, :, :]
        b_star = lab_img[:, 2, :, :]

        # Calculate H and S
        H_lab = torch.atan2(b_star, a_star) / (2 * pi)  # Color angle (Hue)
        S_lab = torch.sqrt(a_star ** 2 + b_star ** 2)  # Saturation

        # Normalize H and S to [0, 1]
        H_lab = H_lab / (2 * pi)
        S_lab = S_lab / (S_lab.max() + 1e-8)

        # You can add a fusion mechanism here to combine these with existing H and S from HVI

        return H_lab, S_lab, a_star, b_star

    def color_contrast(self, a, b, kernel_size=5):
        pad = kernel_size // 2
        a_mean = F.avg_pool1d(a, kernel_size, stride=1, padding=pad)
        b_mean = F.avg_pool1d(b, kernel_size, stride=1, padding=pad)

        diff = (a - a_mean)**2 + (b - b_mean)**2
        C_ab = torch.sqrt(diff + 1e-6)

        C_ab = (C_ab - C_ab.min()) / (C_ab.max() - C_ab.min() + 1e-6 )
        return C_ab.unsqueeze(1)

    def forward(self, rgb_img, lab_img):

        H_lab, S_lab, a, b = self.lab_to_uvi(lab_img.cuda())  # Convert Lab to H and S
        hvi_features = self.UVIT(rgb_img, H_lab, S_lab, a, b)

        return hvi_features