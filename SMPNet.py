import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet


class SE_block(nn.Module):
    def __init__(self, in_channel, ratio):
        super(SE_block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channel, in_channel//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channel//ratio, in_channel, 1, 1, 0)
    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return torch.sigmoid(out)


class JAFF_module(nn.Module):
    def __init__(self, in_channel, se_ratio):
        super(JAFF_module, self).__init__()
        self.ds_conv = self.DS_Conv(in_channel, 2 * in_channel, 2)
        self.se_block = SE_block(in_channel, se_ratio)

    def forward(self, u_v, u_i):
        u_temp = torch.cat((u_v, u_i), dim=1)
        wm_temp = self.ds_conv(u_temp)
        wm = F.softmax(wm_temp, dim=3)
        u_temp2 = torch.cat((torch.mul(u_v, wm[:, 0:1, :, :]), torch.mul(u_i, wm[:, 1:2, :, :])),dim=1)
        cw = self.se_block(u_temp2)
        u = torch.mul(u_temp2,cw)
        return u

    def DS_Conv(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1,groups=in_channel),
            nn.Conv2d(in_channel, intermediate_channel, 1, 1, 0,groups=1),
            nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1,groups=intermediate_channel),
            nn.Conv2d(intermediate_channel, out_channel, 1, 1, 0,groups=1)
        )

class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

class TOFR_module_first(nn.Module):
    def __init__(self, in_channel, pad_pool = (1,0)):
        super(TOFR_module_first, self).__init__()
        self.pam = PositionAttentionModule(in_channel)
        self.cam = ChannelAttentionModule()

        self.conv = nn.Conv2d(in_channel, in_channel * 2, 3,1,1)
        self.bn2 = nn.BatchNorm2d(in_channel*2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2,padding=pad_pool)

    def forward(self, x):
        x = self.pam(x)
        x = self.cam(x)
        out = self.conv(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

    def DS_Conv(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1,groups=in_channel),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0,groups=1)
        )

class TOFR_module(nn.Module):
    def __init__(self, in_channel, pad_pool = (1,0)):
        super(TOFR_module, self).__init__()
        self.pam = PositionAttentionModule(in_channel)
        self.cam = ChannelAttentionModule()

        self.ds_conv = self.DS_Conv(in_channel, in_channel)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channel, in_channel * 2, 3,1,1)
        self.bn2 = nn.BatchNorm2d(in_channel*2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2,padding=pad_pool)

    def forward(self, x, y):
        x = self.pam(x)
        x = self.cam(x)

        y = self.ds_conv(y)
        y = self.bn1(y)
        y = self.sigmoid(y)

        out = x + y
        out = self.conv(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

    def DS_Conv(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1,groups=in_channel),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0,groups=1)
        )


class SMPNet(nn.Module):
    def __init__(self):
        super(SMPNet, self).__init__()
        se_ratio = 4
        ch = [128, 256, 512, 1024]
        self.tasks = {'hr': 1, 'pulse': 250, 'rr': 1}

        backbone_v = resnet.__dict__['resnet34'](pretrained=False)
        backbone_i = resnet.__dict__['resnet34'](pretrained=False)

        self.shared_conv_v = nn.Sequential(backbone_v.conv1, backbone_v.bn1, backbone_v.relu1, backbone_v.maxpool)
        self.shared_conv_i = nn.Sequential(backbone_i.conv1, backbone_i.bn1, backbone_i.relu1, backbone_i.maxpool)


        # 特征共享层
        self.shared_layer1_v = backbone_v.layer1
        self.shared_layer1_i = backbone_i.layer1

        self.shared_layer2_v = backbone_v.layer2
        self.shared_layer2_i = backbone_i.layer2

        self.shared_layer3_v = backbone_v.layer3
        self.shared_layer3_i = backbone_i.layer3

        self.shared_layer4_v = backbone_v.layer4
        self.shared_layer4_i = backbone_i.layer4

        self.ff_block_1 = JAFF_module(ch[0], se_ratio)
        self.ff_block_2 = JAFF_module(ch[1], se_ratio)
        self.ff_block_3 = JAFF_module(ch[2], se_ratio)
        self.ff_block_4 = JAFF_module(ch[3], se_ratio)


        self.modules_tofr_1 = nn.ModuleList([TOFR_module_first(ch[0], pad_pool=(1,1)) for _ in self.tasks])
        self.modules_tofr_2 = nn.ModuleList([TOFR_module(ch[1]) for _ in self.tasks])
        self.modules_tofr_3 = nn.ModuleList([TOFR_module(ch[2],pad_pool=(0,0)) for _ in self.tasks])
        self.modules_tofr_4 = nn.ModuleList([TOFR_module(ch[3]) for _ in self.tasks])


        self.down_sampling_task = nn.ModuleList([nn.AdaptiveMaxPool2d((1,1)) for _ in self.tasks])
        # self.linear = nn.ModuleList([nn.Linear(ch[3],self.tasks[t]) for t in self.tasks])
        self.conv_task = nn.ModuleList([nn.Conv1d(ch[3]*2, self.tasks[t], 1) for t in self.tasks])

    def forward(self, x, y):
        x_v = self.shared_conv_v(x)
        x_i = self.shared_conv_i(y)

        u_1_v = self.shared_layer1_v(x_v)
        u_1_i = self.shared_layer1_i(x_i)

        u_2_v = self.shared_layer2_v(u_1_v)
        u_2_i = self.shared_layer2_i(u_1_i)

        u_3_v = self.shared_layer3_v(u_2_v)
        u_3_i = self.shared_layer3_i(u_2_i)

        u_4_v = self.shared_layer4_v(u_3_v)
        u_4_i = self.shared_layer4_i(u_3_i)

        self.u_1 = self.ff_block_1(u_1_v, u_1_i)
        # print(self.u_1.shape)
        self.u_2 = self.ff_block_2(u_2_v, u_2_i)
        # print(self.u_2.shape)
        self.u_3 = self.ff_block_3(u_3_v, u_3_i)
        # print(self.u_3.shape)
        self.u_4 = self.ff_block_4(u_4_v, u_4_i)
        # print(self.u_4.shape)

        a_1 = [tofr1(self.u_1) for tofr1 in self.modules_tofr_1]
        # print('a_1[0].shape:', a_1[0].shape)

        a_2 = [tofr2(self.u_2, a_1_i) for a_1_i, tofr2 in zip(a_1, self.modules_tofr_2)]
        # print('a_2[0].shape:', a_2[0].shape)

        a_3 = [tofr3(self.u_3, a_2_i) for a_2_i, tofr3 in zip(a_2, self.modules_tofr_3)]
        # print('a_3[0].shape:', a_3[0].shape)

        a_4 = [tofr4(self.u_4, a_3_i) for a_3_i, tofr4 in zip(a_3, self.modules_tofr_4)]
        # print('a_4[0].shape:', a_4[0].shape)

        a_out = [dst_i(a_4_i) for dst_i, a_4_i in zip(self.down_sampling_task, a_4)]
        a_out = [a_out_i.squeeze(-1) for a_out_i in a_out]
        # print('a_out[0].shape:',a_out[0].shape)

        out = [conv_i(a_out_i).squeeze(-1) for conv_i, a_out_i in zip(self.conv_task, a_out)]

        return out[0],out[1],out[2]


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    model=SMPNet()
    x1 = torch.rand(1, 3,276, 250)
    x2 = torch.rand(1, 3, 276, 250)
    y1,y2,y3 = model(x1,x2)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    # print_network(model)
