import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_attn(q, k, x, EPS=1e-7):
    l1, d1 = q.shape[-2:]
    l2, d2 = x.shape[-2:]
    k = k.transpose(-2, -1)
    if l1 * d1 * l2 + l1 * l2 * d2 <= d2 * l2 * d1 + d2 * d1 * l1:
        q = q @ k
        q = q / (q.sum(dim=-1, keepdim=True) + EPS)
        x = q @ x
    else:
        x = q @ (k @ x)
        q = q @ k.sum(dim=-1, keepdim=True) + EPS
        x = x / q
    return x


class MultiheadAttetion(nn.Module):
    def __init__(self, convDim, numHeads, patchSize, qkScale=None, qkvBias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.convDim = convDim
        self.numHeads = numHeads
        self.patchSize = patchSize
        self.relu = nn.ReLU()
        self.register_buffer('one', None)
        self.qkv = nn.Conv2d(self.convDim, self.convDim * 3, 1, 1, 0)
        self.proj = nn.Conv2d(self.convDim, self.convDim, 3, 1, 1)

    def forward(self, x, mask=None, padsize=0):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.convDim, H, W).transpose(0, 1)
        q, k, x = qkv[0], qkv[1], qkv[2]
        del qkv
        q, k = self.relu(q), self.relu(k)
        q = F.unfold(q, self.patchSize, padding=(self.patchSize - 1) // 2, stride=1)  # [B, numHead, headDim, H, W]
        k = F.unfold(k, self.patchSize, padding=(self.patchSize - 1) // 2, stride=1)
        x = x.view(B, C, H * W)
        if q.shape[1] % self.numHeads != 0:
            padsize = (q.shape[1] // self.numHeads + 1) * self.numHeads - q.shape[1]
            q = F.pad(q, [0, 0, 0, padsize], mode='replicate')
            k = F.pad(k, [0, 0, 0, padsize], mode='replicate')
            x = F.pad(x, [0, 0, 0, padsize], mode='replicate')
        q = q.view(B, self.numHeads, -1, H * W).transpose(-1, -2)
        k = k.view(B, self.numHeads, -1, H * W).transpose(-1, -2)
        x = x.view(B, self.numHeads, -1, H * W).transpose(-1, -2)
        x = linear_attn(q, k, x)
        x = x.transpose(-1, -2).contiguous().view(B, -1, H * W)
        if padsize != 0:
            x = x[:, :-padsize, :]
        x = x.view(B, C, H, W)
        x = self.proj(x)
        return x

    def flops(self, x):
        B, C, H, W = x.shape
        d1 = C * self.patchSize ** 2  # for q and k
        l1 = H * W
        d2 = C
        flops = 0
        flops += l1 * C * 3 * C
        flops += min(l1 * d1 * l1 + l1 * l1 * d2, l1 * d1 * d2 + d1 * l1 * d2)
        flops += l1 * C * 9 * C  #
        return flops


class speMultiAttn(nn.Module):
    def __init__(self, convDim, numHeads=8, poolSize=4, ksize=9) -> None:
        super().__init__()
        self.numHeads = numHeads
        self.convDim = convDim
        self.poolSize = poolSize
        self.avepool = nn.AdaptiveAvgPool2d(self.poolSize)
        self.relu = nn.ReLU()
        self.q = nn.Linear(self.poolSize ** 2, self.poolSize ** 2 * self.numHeads)
        self.k = nn.Linear(self.poolSize ** 2, self.poolSize ** 2 * self.numHeads)
        self.v = nn.Conv2d(self.convDim, self.convDim, 1, 1, 0)
        self.proj = nn.Conv2d(self.convDim, self.convDim, ksize, 1, (ksize - 1) // 2, groups=self.convDim)

    def forward(self, x, padsize=0):
        B, C, H, W = x.shape
        q = self.avepool(x)
        q = q.view(B, C, -1)
        q = self.q(q)
        k = self.avepool(x)
        k = k.view(B, C, -1)
        k = self.k(k)
        x = self.v(x)
        q = self.relu(q).view(B, C, self.numHeads, -1)
        k = self.relu(k).view(B, C, self.numHeads, -1)

        x = x.view(B, C, -1)
        if x.shape[2] % self.numHeads != 0:
            padsize = (x.shape[2] // self.numHeads + 1) * self.numHeads - x.shape[2]
            x = F.pad(x, [0, padsize, 0, 0], mode='replicate')
        x = x.view(B, C, self.numHeads, -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        x = x.transpose(1, 2)
        x = linear_attn(q, k, x)
        x = x.transpose(1, 2).contiguous().view(B, C, -1)
        if padsize:
            x = x[:, :, :-padsize]
        x = x.view(B, C, H, W)
        x = self.proj(x)
        return x


class spaTransBlock(nn.Module):
    def __init__(self, convDim):
        super().__init__()
        self.multiAttn = MultiheadAttetion(convDim, 8, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(convDim, convDim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(convDim, convDim, 3, 1, 1, groups=convDim),
            nn.GELU(),
            nn.Conv2d(convDim, convDim, 1, 1, 0),
        )
        self.reweight = nn.Parameter(torch.zeros(1)) if False else 1

    def forward(self, x):
        x = x + self.multiAttn(x) * self.reweight
        x = x + self.ffn(x) * self.reweight
        return x


class speTransBlock(nn.Module):
    def __init__(self, convDim):
        super().__init__()
        self.multiattn = speMultiAttn(convDim)
        self.ffn = nn.Sequential(
            nn.Conv2d(convDim, convDim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(convDim, convDim, 3, 1, 1, groups=convDim),
            nn.GELU(),
            nn.Conv2d(convDim, convDim, 1, 1, 0),
        )
        self.reweight = nn.Parameter(torch.zeros(1)) if False else 1

    def forward(self, x):
        x = x + self.multiattn(x) * self.reweight
        x = x + self.ffn(x) * self.reweight
        return x


class ConvUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(ConvUnit, self).__init__()
        if kernel_size == 3:
            self.basic_unit = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.ReLU(True),
                nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                          bias=False),
            )
        else:
            self.basic_unit = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1,
                          bias=False),
                nn.ReLU(True),
                nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1,
                          bias=False),
            )

    def forward(self, input):
        return self.basic_unit(input)


class Get_Grad(nn.Module):
    def __init__(self, in_channels=1):
        super(Get_Grad, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        nn.init.constant_(self.conv1.weight, -1)
        for i in range(in_channels):
            for j in range(in_channels):
                nn.init.constant_(self.conv1.weight[j, i, 1, 1], 8)
        for param in self.conv1.parameters():
            param.requires_grad = False

    def forward(self, x1):
        edge_map = self.conv1(x1)
        return edge_map


class CFI(nn.Module):
    def __init__(self, xs_channel=4, xg_channel=4, n_feat=64):
        super(CFI, self).__init__()
        self.xg_channel = xg_channel
        self.xs_channel = xs_channel
        self.fu = ConvUnit(in_channels=xg_channel + xs_channel, out_channels=xg_channel + xs_channel,
                           mid_channels=n_feat)
        self.trans1 = ConvUnit(in_channels=xs_channel * 2, out_channels=xs_channel, mid_channels=n_feat)
        self.trans2 = ConvUnit(in_channels=xg_channel * 2, out_channels=xg_channel, mid_channels=n_feat)

    def forward(self, xs, xg):
        tmp = torch.cat([xs, xg], dim=1)
        tmp = self.fu(tmp)
        xs = xs - self.trans1(torch.cat([xs, tmp[:, :self.xs_channel, :, :]], dim=1))
        xg = xg - self.trans2(torch.cat([xg, tmp[:, -self.xg_channel:, :, :]], dim=1))
        del tmp
        return xs, xg


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class Ms_fusion(nn.Module):
    def __init__(self, channel=1):
        super(Ms_fusion, self).__init__()
        mid_channels = channel * 2
        out_channels = channel * 4
        self.Ps1 = nn.Sequential(
            nn.Conv2d(channel, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.Ps2 = nn.Sequential(
            nn.Conv2d(channel * 2, out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.Ps3 = nn.Sequential(
            nn.Conv2d(channel * 3, out_channels * 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.scale = 2

    def forward(self, f1, f2, f3, f4):
        f4 = self.Ps1(f4)
        f34 = torch.cat([f3, f4], dim=1)
        f34 = self.Ps2(f34)
        f234 = torch.cat([f2, f34], dim=1)
        f234 = self.Ps3(f234)
        f1234 = torch.cat([f1, f234], dim=1)
        return f1234


class DownModel(nn.Module):  # 下采样模块
    def __init__(self, H=128):
        super(DownModel, self).__init__()
        self.donw1 = nn.AdaptiveMaxPool2d((H // 2, H // 2))
        self.donw2 = nn.AdaptiveMaxPool2d((H // 4, H // 4))
        self.donw3 = nn.AdaptiveMaxPool2d((H // 8, H // 8))

    def forward(self, inputs):
        output1 = self.donw1(inputs)
        output2 = self.donw2(inputs)
        output3 = self.donw3(inputs)
        return inputs, output1, output2, output3


class UpModel(nn.Module):
    def __init__(self, ms_channels=4, H=128):
        super(UpModel, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(ms_channels, ms_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ms_channels * 2, ms_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(ms_channels, ms_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ms_channels * 4, ms_channels * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(4)
        )
        self.donw1 = nn.AdaptiveMaxPool2d((H // 8, H // 8))

    def forward(self, inputs):
        output1 = self.up1(inputs)
        output2 = self.up2(inputs)
        output3 = self.donw1(inputs)
        return output2, output1, inputs, output3


class DPN1(nn.Module):
    def __init__(self, ms_channels=4, n_feat=64, pan_channels=1, kernel_size=3, xs_channel=4):
        super(DPN1, self).__init__()
        self.down = DownModel()
        self.up = UpModel(ms_channels=ms_channels)
        self.conv1 = ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=ms_channels, kernel_size=1)
        self.conv2 = ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=ms_channels, kernel_size=1)
        self.conv3 = ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=ms_channels, kernel_size=1)
        self.conv4 = ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=ms_channels, kernel_size=1)
        self.fusion = Ms_fusion(channel=ms_channels)

        self.conv11 = ConvUnit(in_channels=ms_channels * 4, mid_channels=n_feat, out_channels=xs_channel,
                               kernel_size=3)
        self.spe = speTransBlock(convDim=xs_channel)

    def forward(self, HR, LR):
        _, _, M, N = HR.shape
        _, _, m, n = LR.shape
        h1, h2, h3, h4 = self.down(HR)  # 由大到小
        l1, l2, l3, l4 = self.up(LR)  # 由大到小
        ms_Residual1, ms_Residual2, ms_Residual3, ms_Residual4 = self.conv1(h1 - l1), self.conv2(h2 - l2), self.conv3(
            h3 - l3), self.conv4(h4 - l4)
        xs = self.fusion(ms_Residual1, ms_Residual2, ms_Residual3, ms_Residual4)
        xs = self.conv11(xs)
        xs = self.spe(xs)

        return xs


class DPN2(nn.Module):
    def __init__(self, ms_channels, n_feat, pan_channels, xg_channel):
        super(DPN2, self).__init__()
        self.LAP = Get_Grad(in_channels=pan_channels)
        self.down1 = DownModel()
        self.down2 = DownModel()
        self.get_PAN1 = ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=pan_channels)
        self.get_PAN2 = ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=pan_channels)
        self.get_PAN3 = ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=pan_channels)
        self.fusion = Ms_fusion(channel=pan_channels)
        self.conv11 = ConvUnit(in_channels=pan_channels * 4, mid_channels=n_feat, out_channels=xg_channel,
                               kernel_size=1)
        self.spa = spaTransBlock(convDim=xg_channel)

    def forward(self, HR, PAN):
        H1, H2, H3, H4 = self.down1(HR)
        P1, P2, P3, P4 = self.down2(PAN)
        dP1, dP2, dP3, dP4 = self.LAP(P1), self.LAP(P2), self.LAP(P3), self.LAP(P4)
        _dP1, _dP2, _dP3, _dP4 = self.LAP(self.get_PAN1(H1)), self.LAP(self.get_PAN2(H2)), self.LAP(
            self.get_PAN3(H3)), self.LAP(self.get_PAN3(H4))
        dPAN_Residual1, dPAN_Residual2, dPAN_Residual3, dPAN_Residual4 = dP1 - _dP1, dP2 - _dP2, dP3 - _dP3, dP4 - _dP4
        xg = self.fusion(dPAN_Residual1, dPAN_Residual2, dPAN_Residual3, dPAN_Residual4)
        xg = self.conv11(xg)
        xg = self.spa(xg)
        return xg


class prox(nn.Module):
    def __init__(self, ms_channels=4, n_feat=64, pan_channels=1, kernel_size=3, H=128, xs_channel=4, xg_channel=3):
        super(prox, self).__init__()
        self.AT = nn.Sequential(
            ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=ms_channels, kernel_size=3),
            nn.Upsample(size=(H, H), mode='bicubic')
        )
        self.A = nn.Sequential(
            ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=ms_channels, kernel_size=3),
            nn.Upsample(size=(H // 4, H // 4), mode='bicubic')
        )
        self.CT = ConvUnit(in_channels=pan_channels, mid_channels=n_feat, out_channels=ms_channels, kernel_size=1)
        self.C = ConvUnit(in_channels=ms_channels, mid_channels=n_feat, out_channels=pan_channels, kernel_size=1)
        self.DPN1T = ConvUnit(in_channels=xs_channel, mid_channels=n_feat, out_channels=ms_channels, kernel_size=3)
        self.DPN2T = ConvUnit(in_channels=xg_channel, out_channels=ms_channels, mid_channels=n_feat, kernel_size=3)

        self.u1 = torch.nn.Parameter(torch.ones(1, ms_channels, 1, 1) * 0.5)
        self.u2 = torch.nn.Parameter(torch.ones(1, ms_channels, 1, 1) * 0.5)

    def forward(self, H0, ms, pan, d11, d21, b11, b21, xs, xg):
        dH = self.u1 * self.AT(self.A(H0) - ms) + (1 - self.u1) * self.CT(self.C(H0) - pan) + \
             self.u2 * self.DPN1T(xs - d11 + b11) + (1 - self.u2) * self.DPN2T(xg - d21 + b21)
        return dH


class one_step(nn.Module):
    def __init__(self,
                 ms_channels, pan_channels, xg_channel, xs_channel, n_feat):
        super(one_step, self).__init__()
        self.n_feat = n_feat
        self.DPN1 = DPN1(ms_channels=ms_channels, pan_channels=pan_channels, n_feat=self.n_feat, xs_channel=xs_channel)
        self.DPN2 = DPN2(ms_channels=ms_channels, pan_channels=pan_channels, n_feat=self.n_feat, xg_channel=xg_channel)
        self.u1 = torch.nn.Parameter(torch.ones(1, ms_channels, 1, 1) * 0.0001)
        self.u2 = torch.nn.Parameter(torch.ones(1, ms_channels, 1, 1) * 0.0001)
        self.a = torch.nn.Parameter(torch.ones(1, ms_channels, 1, 1))
        self.prox = prox(ms_channels=ms_channels, pan_channels=pan_channels, xg_channel=xg_channel,
                         xs_channel=xs_channel, n_feat=self.n_feat)
        self.CFI = CFI(xs_channel=xs_channel, xg_channel=xg_channel, n_feat=self.n_feat)

    def softshrink(self, x, mu):
        z = torch.zeros_like(x)
        x = torch.sign(x) * torch.maximum(torch.abs(x) - mu, z)
        return x

    def forward(self, H0, ms, pan, b1, b2, d1, d2):
        xs = self.DPN1(H0, ms)
        xg = self.DPN2(H0, pan)
        xs, xg = self.CFI(xs, xg)
        d11 = self.softshrink(xs + b1, self.u1)
        d21 = self.softshrink(xg + b2, self.u2)
        b11 = b1 + xs - d1
        b21 = b2 + xg - d2
        H1 = H0 + self.a * self.prox(H0, ms, pan, d11, d21, b11, b21, xs, xg)
        return H1, b11, b21, d11, d21, xs, xg


class PS_unfolding(nn.Module):
    def __init__(self,
                 ms_channels=4,
                 pan_channels=1,
                 stage=11,
                 n_feat=32):
        super(PS_unfolding, self).__init__()
        self.n_feat = n_feat
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.xg_channels = ms_channels
        self.xs_channels = ms_channels
        self.stage = stage
        self.iters = nn.ModuleList()
        for i in range(self.stage):
            self.iters.append(self.make_step())
        self.dense_fuse = ConvUnit(in_channels=self.ms_channels * self.stage, out_channels=self.ms_channels,
                                   mid_channels=self.n_feat)

    def make_step(self):
        return one_step(ms_channels=self.ms_channels, pan_channels=self.pan_channels, xs_channel=self.xs_channels,
                        xg_channel=self.xg_channels, n_feat=self.n_feat)

    def forward(self, ms, pan):
        b, c, h, w = pan.shape
        H0 = upsample(ms, h, w)
        b, c, h, w = H0.shape
        H_menmery = []
        d1 = torch.zeros((b, self.xs_channels, h, w), requires_grad=False).cuda()
        d2 = torch.zeros((b, self.xg_channels, h, w), requires_grad=False).cuda()
        b1 = torch.zeros_like(d1, requires_grad=False).cuda()
        b2 = torch.zeros_like(d2, requires_grad=False).cuda()

        for i in range(self.stage):
            H0, b1, b2, d1, d2, xs, xg = self.iters[i](H0, ms, pan, b1, b2, d1, d2)
            H_menmery.append(H0)
        tmp = self.dense_fuse(torch.cat(H_menmery, dim=1))
        H0 = H0 + tmp
        return H0


class Loss_Function(nn.Module):
    def __init__(self, s=0.1):
        super(Loss_Function, self).__init__()
        self.lossl1 = nn.L1Loss()
        self.get_grad = Get_Grad(in_channels=4)
        self.s = s

    def forward(self, out, target):
        H = out
        target = target.cuda()
        loss = self.lossl1(H, target)
        loss2 = self.lossl1(self.get_grad(H), self.get_grad(target))
        return loss + self.s * torch.log(loss2)


if __name__ == '__main__':
    net = PS_unfolding().cuda()
    ms = torch.rand(4, 4, 32, 32).cuda()
    pan = torch.rand(4, 1, 128, 128).cuda()
    out = net(ms, pan)
    print(out.shape)
