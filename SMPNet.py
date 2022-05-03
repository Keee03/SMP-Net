import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet

class ROIA(nn.Module):
    def __init__(self, num_roi, num_c):
        super(ROIA, self).__init__()
        self.layer = nn.ModuleList([nn.Conv1d(num_c, num_c, 3, 1, 1) for _ in range(num_roi)])
        self.conv_roi = nn.ModuleList([nn.Conv1d(num_c, 1, 3, 1, 1) for _ in range(num_roi)])
        self.pool_roi = nn.AdaptiveAvgPool2d((num_roi, 1))
        self.conv_c = nn.Conv2d(num_c, num_c, 3)
        self.pool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_new = [self.layer[i](x[:, :, i, :]) for i in range(x.shape[2])]
        x_roi = [self.conv_roi[i](x_new[i]) for i in range(len(x_new))]
        x_new = torch.stack(x_new).permute(1, 2, 0, 3)
        x_roi = torch.stack(x_roi).permute(1, 2, 0, 3)
        w_roi = self.sigmoid(self.pool_roi(x_roi))
        x_c = self.conv_c(x_new)
        w_c = self.sigmoid(self.pool_c(x_c))
        out = torch.mul(torch.mul(x, w_roi), w_c)
        return out

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

class FF_block(nn.Module):
    def __init__(self, in_channel, se_ratio):
        super(FF_block, self).__init__()
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

class SMPNet(nn.Module):
    def __init__(self):
        super(SMPNet, self).__init__()
        se_ratio = 4
        # ch = [64, 128, 256, 512]
        ch = [128, 256, 512, 1024]
        self.tasks = {'hr': 1, 'pulse': 250, 'rr': 1}

        backbone_v = resnet.__dict__['resnet34'](pretrained=False)
        backbone_i = resnet.__dict__['resnet34'](pretrained=False)

        self.roia_block_v = ROIA(276, 3)
        self.roia_block_i = ROIA(276, 3)

        self.shared_conv_v = nn.Sequential(backbone_v.conv1, backbone_v.bn1, backbone_v.relu1, backbone_v.maxpool)
        self.shared_conv_i = nn.Sequential(backbone_i.conv1, backbone_i.bn1, backbone_i.relu1, backbone_i.maxpool)


        # 特征共享层
        self.shared_layer1_v_b = backbone_v.layer1[:-1]
        self.shared_layer1_v_t = backbone_v.layer1[-1]
        self.shared_layer1_i_b = backbone_i.layer1[:-1]
        self.shared_layer1_i_t = backbone_i.layer1[-1]

        self.shared_layer2_v_b = backbone_v.layer2[:-1]
        self.shared_layer2_v_t = backbone_v.layer2[-1]
        self.shared_layer2_i_b = backbone_i.layer2[:-1]
        self.shared_layer2_i_t = backbone_i.layer2[-1]

        self.shared_layer3_v_b = backbone_v.layer3[:-1]
        self.shared_layer3_v_t = backbone_v.layer3[-1]
        self.shared_layer3_i_b = backbone_i.layer3[:-1]
        self.shared_layer3_i_t = backbone_i.layer3[-1]

        self.shared_layer4_v_b = backbone_v.layer4[:-1]
        self.shared_layer4_v_t = backbone_v.layer4[-1]
        self.shared_layer4_i_b = backbone_i.layer4[:-1]
        self.shared_layer4_i_t = backbone_i.layer4[-1]

        self.ff_block_1b = FF_block(ch[0], se_ratio)
        self.ff_block_1t = FF_block(ch[0], se_ratio)
        self.ff_block_2b = FF_block(ch[1], se_ratio)
        self.ff_block_2t = FF_block(ch[1], se_ratio)
        self.ff_block_3b = FF_block(ch[2], se_ratio)
        self.ff_block_3t = FF_block(ch[2], se_ratio)
        self.ff_block_4b = FF_block(ch[3], se_ratio)
        self.ff_block_4t = FF_block(ch[3], se_ratio)

        # 接收前一层的任务特征以及共享特征1计算得到mask
        self.encoder_mask_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_mask_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_mask_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_mask_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # mask与共享特征2相乘后，经过此块编码后传递到下一层
        self.encoding_block_1 = nn.ModuleList([self.encoding_layer(ch[0], ch[1]) for _ in self.tasks])
        self.encoding_block_2 = nn.ModuleList([self.encoding_layer(ch[1], ch[2]) for _ in self.tasks])
        self.encoding_block_3 = nn.ModuleList([self.encoding_layer(ch[2], ch[3]) for _ in self.tasks])

        self.down_sampling_1 = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 1)) for _ in self.tasks])
        self.down_sampling_2 = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 0)) for _ in self.tasks])
        self.down_sampling_3 = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in self.tasks])

        self.down_sampling_task = nn.ModuleList([nn.AdaptiveMaxPool2d((1,1)) for _ in self.tasks])
        # self.linear = nn.ModuleList([nn.Linear(ch[3],self.tasks[t]) for t in self.tasks])
        self.conv_task_1 = nn.ModuleList([nn.Conv1d(ch[3], ch[3]*2, 1) for t in self.tasks])
        self.conv_task_2 = nn.ModuleList([nn.Conv1d(ch[3]*2, self.tasks[t], 1) for t in self.tasks])

    def forward(self, x, y):
        x_v = self.roia_block_v(x)
        x_i = self.roia_block_i(y)
        x_v = self.shared_conv_v(x_v)
        x_i = self.shared_conv_i(x_i)


        u_1_v_b = self.shared_layer1_v_b(x_v)
        u_1_v_t = self.shared_layer1_v_t(u_1_v_b)
        u_1_i_b = self.shared_layer1_i_b(x_i)
        u_1_i_t = self.shared_layer1_i_t(u_1_i_b)
        # print('u_1_b.shape:', u_1_b.shape)
        # print('u_1_t.shape:', u_1_t.shape)

        u_2_v_b = self.shared_layer2_v_b(u_1_v_t)
        u_2_v_t = self.shared_layer2_v_t(u_2_v_b)
        u_2_i_b = self.shared_layer2_i_b(u_1_i_t)
        u_2_i_t = self.shared_layer2_i_t(u_2_i_b)
        # print('u_2_b.shape:',u_2_b.shape)
        # print('u_2_t.shape:',u_2_t.shape)

        u_3_v_b = self.shared_layer3_v_b(u_2_v_t)
        u_3_v_t = self.shared_layer3_v_t(u_3_v_b)
        u_3_i_b = self.shared_layer3_i_b(u_2_i_t)
        u_3_i_t = self.shared_layer3_i_t(u_3_i_b)
        # print('u_3_b.shape:', u_3_b.shape)
        # print('u_3_t.shape:', u_3_t.shape)

        u_4_v_b = self.shared_layer4_v_b(u_3_v_t)
        u_4_v_t = self.shared_layer4_v_t(u_4_v_b)
        u_4_i_b = self.shared_layer4_i_b(u_3_i_t)
        u_4_i_t = self.shared_layer4_i_t(u_4_i_b)
        # print('u_4_b.shape:', u_4_b.shape)
        # print('u_4_t.shape:', u_4_t.shape)

        self.u_1_b = self.ff_block_1b(u_1_v_b, u_1_i_b)
        self.u_1_t = self.ff_block_1t(u_1_v_t, u_1_i_t)
        self.u_2_b = self.ff_block_2b(u_2_v_b, u_2_i_b)
        self.u_2_t = self.ff_block_2t(u_2_v_t, u_2_i_t)
        self.u_3_b = self.ff_block_3b(u_3_v_b, u_3_i_b)
        self.u_3_t = self.ff_block_3t(u_3_v_t, u_3_i_t)
        self.u_4_b = self.ff_block_4b(u_4_v_b, u_4_i_b)
        self.u_4_t = self.ff_block_4t(u_4_v_t, u_4_i_t)


        a_1_mask = [att_i(self.u_1_b) for att_i in self.encoder_mask_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * self.u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [enblock1_i(a_1_i) for  a_1_i,enblock1_i in zip(a_1,self.encoding_block_1)]
        a_1 = [ds1_i(a_1_i) for a_1_i,ds1_i in zip(a_1,self.down_sampling_1)]
        # print('a_1[0].shape:',a_1[0].shape)

        a_2_mask = [att_i(torch.cat((self.u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_mask_2)]
        a_2 = [a_2_mask_i * self.u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [enblock2_i(a_2_i) for a_2_i, enblock2_i in zip(a_2, self.encoding_block_2)]
        a_2 = [ds2_i(a_2_i) for  a_2_i,ds2_i in zip(a_2,self.down_sampling_2)]
        # print('a_2[0].shape:',a_2[0].shape)

        a_3_mask = [att_i(torch.cat((self.u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_mask_3)]
        a_3 = [a_3_mask_i * self.u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [enblock3_i(a_3_i) for a_3_i, enblock3_i in zip(a_3, self.encoding_block_3)]
        a_3 = [ds3_i(a_3_i) for a_3_i, ds3_i in zip(a_3, self.down_sampling_3)]
        # print('a_3[0].shape:',a_3[0].shape)

        a_4_mask = [att_i(torch.cat((self.u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_mask_4)]
        a_4 = [a_4_mask_i * self.u_4_t for a_4_mask_i in a_4_mask]
        # print('a_4[0].shape:',a_4[0].shape)

        a_out=[dst_i(a_4_i) for dst_i,a_4_i in zip(self.down_sampling_task,a_4)]
        a_out=[a_out_i.squeeze(-1) for a_out_i in a_out]
        # print('a_out[0].shape:', a_out[0].shape)

        # out = [lieanr_i(out_i) for lieanr_i, out_i in zip(self.linear, a_out)]
        out = [conv1_i(out_i) for conv1_i, out_i in zip(self.conv_task_1, a_out)]
        out = [conv2_i(out_i).squeeze(-1) for conv2_i, out_i in zip(self.conv_task_2, out)]

        return out[0],out[1],out[2]

    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
            )

    def encoding_layer(self,in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    model=SMPNet()
    x1 = torch.rand(1, 3,276, 250)
    x2 = torch.rand(1, 3, 276, 250)
    y1,y2,y3=model(x1,x2)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    print_network(model)