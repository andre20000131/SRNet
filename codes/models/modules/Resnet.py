import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.cuda.is_available())


class Resblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mode='down'):
        super(Resblock, self).__init__()

        # self.conv_body_first = nn.Conv2d(3, 32, 1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2

    def forward(self, x):
        # feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)

        # xunhuan 0
        out = F.relu(self.conv1(x))
        out = F.interpolate(out, scale_factor=self.scale_factor, mode='nearest')
        out = F.relu(self.conv2(out))

        # skip
        feat_skip = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        skip = self.skip(feat_skip)
        out = out + skip
        return out


class SRNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(SRNet, self).__init__()
        self.conv_body_first = nn.Conv2d(3, 32, 1)
        self.conv_down0 = Resblock(32, 64, mode='down')
        self.conv_down1 = Resblock(64, 128)
        self.conv_down2 = Resblock(128, 256)
        self.conv_down3 = Resblock(256, 256)
        self.conv_down4 = Resblock(256, 256)
        self.conv_down5 = Resblock(256, 256)
        self.conv_down6 = Resblock(256, 256)

        self.conv_up0 = Resblock(256, 256, mode='up')
        self.conv_up1 = Resblock(256, 256, mode='up')
        self.conv_up2 = Resblock(256, 256, mode='up')
        self.conv_up3 = Resblock(256, 256, mode='up')
        self.conv_up4 = Resblock(256, 128, mode='up')
        self.conv_up5 = Resblock(128, 64, mode='up')
        self.conv_up6 = Resblock(64, 32, mode='up')

        self.conv_condition_scale0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_condition_scale1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_condition_scale2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_condition_scale3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_condition_scale4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv_condition_scale5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_condition_scale6 = nn.Conv2d(32, 32, 3, 1, 1)

        self.conv_condition_shift0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_condition_shift1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_condition_shift2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_condition_shift3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_condition_shift4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv_condition_shift5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_condition_shift6 = nn.Conv2d(32, 32, 3, 1, 1)

        self.final_conv = nn.Conv2d(256, 256, 3, 1, 1, )
        self.out_conv = nn.Conv2d(32,3,3,1,1)
    def forward(self, x):
        unet_skips = []
        condition = []
        feat = F.relu(self.conv_body_first(x))

        feat = self.conv_down0(feat)
        unet_skips.insert(0, feat)

        feat = self.conv_down1(feat)
        unet_skips.insert(0, feat)

        feat = self.conv_down2(feat)
        unet_skips.insert(0, feat)

        feat = self.conv_down3(feat)
        unet_skips.insert(0, feat)

        feat = self.conv_down4(feat)
        unet_skips.insert(0, feat)
        feat = self.conv_down5(feat)
        unet_skips.insert(0, feat)

        feat = self.conv_down6(feat)
        unet_skips.insert(0, feat)

        feat = F.relu(self.final_conv(feat))

        feat = feat + unet_skips[0]
        feat = self.conv_up0(feat)

        #scale = F.relu(self.conv_condition_scale0(feat))
        #condition.append(scale.clone())
        #shift = F.relu(self.conv_condition_shift0(feat))
        #condition.append(shift.clone())

        feat = feat + unet_skips[1]
        feat = self.conv_up1(feat)

        #scale = F.relu(self.conv_condition_scale1(feat))
        #condition.append(scale.clone())
        #shift = F.relu(self.conv_condition_shift1(feat))
        #condition.append(shift.clone())

        feat = feat + unet_skips[2]
        feat = self.conv_up2(feat)

        #scale = F.relu(self.conv_condition_scale2(feat))
        #condition.append(scale.clone())
        #shift = F.relu(self.conv_condition_shift2(feat))
        #condition.append(shift.clone())

        feat = feat + unet_skips[3]
        feat = self.conv_up3(feat)

        #scale = F.relu(self.conv_condition_scale3(feat))
        #condition.append(scale.clone())
        #shift = F.relu(self.conv_condition_shift3(feat))
        #condition.append(shift.clone())

        feat = feat + unet_skips[4]
        feat = self.conv_up4(feat)

        #scale = F.relu(self.conv_condition_scale4(feat))
        #condition.append(scale.clone())
        #shift = F.relu(self.conv_condition_shift4(feat))
        #condition.append(shift.clone())

        feat = feat + unet_skips[5]
        feat = self.conv_up5(feat)

        #scale = F.relu(self.conv_condition_scale5(feat))
        #condition.append(scale.clone())
        #shift = F.relu(self.conv_condition_shift5(feat))
        #condition.append(shift.clone())

        feat = feat + unet_skips[6]
        feat = self.conv_up6(feat)

        out = self.out_conv(feat)
        out = out + x
        return out
        #scale = F.relu(self.conv_condition_scale6(feat))
        #condition.append(scale.clone())
        #shift = F.relu(self.conv_condition_shift6(feat))
        #condition.append(shift.clone())



