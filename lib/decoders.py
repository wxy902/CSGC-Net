import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import typing as t
from einops import rearrange
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv1d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # 默认初始化
            if hasattr(module, 'kernel_size') and len(module.kernel_size) > 0:
                fan_out = module.kernel_size[0] * module.out_channels
                if isinstance(module, nn.Conv2d):
                    fan_out *= module.kernel_size[1]
                fan_out //= module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


# --- 改进后的 DPCF 模块 ---

class AdaptiveCombiner(nn.Module):
    def __init__(self, channels):
        super(AdaptiveCombiner, self).__init__()
        self.d = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, p, i):
        edge_att = torch.sigmoid(self.d)
        return edge_att * p + (1 - edge_att) * i


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class DPCF(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ac = AdaptiveCombiner(in_features // 4)
        self.tail_conv = nn.Sequential(
            conv_block(in_features=in_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0))
        )

    def forward(self, x_low, x_high):
        image_size = x_low.size(2)
        if x_low is not None:
            x_low_chunks = torch.chunk(x_low, 4, dim=1)
        if x_high is not None:
            x_high = F.interpolate(x_high, size=[image_size, image_size], mode='bilinear', align_corners=True)
            x_high_chunks = torch.chunk(x_high, 4, dim=1)

        x0 = self.ac(x_low_chunks[0], x_high_chunks[0])
        x1 = self.ac(x_low_chunks[1], x_high_chunks[1])
        x2 = self.ac(x_low_chunks[2], x_high_chunks[2])
        x3 = self.ac(x_low_chunks[3], x_high_chunks[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False
        self.ex_channels = int(self.in_channels * self.expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )

        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)

        # --- 新增：动态融合模块初始化 ---
        if self.add == True:
            self.fusion_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.ex_channels, max(1, self.ex_channels // 4), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, self.ex_channels // 4), self.ex_channels * self.n_scales, 1, bias=True)
            )
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        # ----------------------------

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)

        # --- 修改：动态融合逻辑 ---
        if self.add == True:
            # 1. 堆叠不同尺度的特征 [B, n_scales, C, H, W]
            stacked_outs = torch.stack(msdc_outs, dim=1)
            # 2. 计算全局池化后的权重 [B, C * n_scales, 1, 1]
            attn = self.fusion_attn(torch.sum(stacked_outs, dim=1))
            # 3. 调整形状以进行逐尺度加权 [B, n_scales, C, 1, 1]
            attn = attn.view(attn.size(0), self.n_scales, self.ex_channels, 1, 1)
            attn = torch.softmax(attn, dim=1)
            # 4. 加权求和 [B, C, H, W]
            dout = torch.sum(stacked_outs * attn, dim=1)
        else:
            dout = torch.cat(msdc_outs, dim=1)
        # ------------------------

        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


# --- 替换后的注意力模块 ---

class Progressive_Channel_wise_Self_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int = 8,
            qkv_bias: bool = False,
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(Progressive_Channel_wise_Self_Attention, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5

        assert self.dim % 4 == 0, '输入特征的维度应能被4整除。'

        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        temp = y
        y = y.mean((2, 3), keepdim=True)
        _, _, h_, w_ = y.size()
        y = self.norm(y)

        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))

        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * temp


class Shareable_Multi_Semantic_Spatial_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            gate_layer: str = 'sigmoid',
    ):
        super(Shareable_Multi_Semantic_Spatial_Attention, self).__init__()
        self.dim = dim
        assert self.dim % 4 == 0, '输入特征的维度应能被4整除。'
        self.group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=self.group_chans)
        self.global_dwc_s = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=self.group_chans)
        self.global_dwc_m = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=self.group_chans)
        self.global_dwc_l = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=self.group_chans)

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = x.size()
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)

        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        return x * x_h_attn * x_w_attn


# --- EMCAD (集成新注意力模块) ---

class EMCAD(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, activation='relu6', lgag_ks=3):
        super(EMCAD, self).__init__()
        eucb_ks = 3

        # 阶段 4
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)
        self.cab4 = Progressive_Channel_wise_Self_Attention(channels[0])
        self.sab4 = Shareable_Multi_Semantic_Spatial_Attention(channels[0])

        # 阶段 3
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.dpcf3 = DPCF(in_features=channels[1], out_features=channels[1])
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)
        self.cab3 = Progressive_Channel_wise_Self_Attention(channels[1])
        self.sab3 = Shareable_Multi_Semantic_Spatial_Attention(channels[1])

        # 阶段 2
        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.dpcf2 = DPCF(in_features=channels[2], out_features=channels[2])
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)
        self.cab2 = Progressive_Channel_wise_Self_Attention(channels[2])
        self.sab2 = Shareable_Multi_Semantic_Spatial_Attention(channels[2])

        # 阶段 1
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.dpcf1 = DPCF(in_features=channels[3], out_features=channels[3])
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)
        self.cab1 = Progressive_Channel_wise_Self_Attention(channels[3])
        self.sab1 = Shareable_Multi_Semantic_Spatial_Attention(channels[3])

    def forward(self, x, skips):
        # 阶段 4
        d4 = self.cab4(x)
        d4 = self.sab4(d4)
        d4 = self.mscb4(d4)

        # 阶段 3
        d3 = self.eucb3(d4)
        d3 = self.dpcf3(skips[0], d3)
        d3 = self.cab3(d3)
        d3 = self.sab3(d3)
        d3 = self.mscb3(d3)

        # 阶段 2
        d2 = self.eucb2(d3)
        d2 = self.dpcf2(skips[1], d2)
        d2 = self.cab2(d2)
        d2 = self.sab2(d2)
        d2 = self.mscb2(d2)

        # 阶段 1
        d1 = self.eucb1(d2)
        d1 = self.dpcf1(skips[2], d1)
        d1 = self.cab1(d1)
        d1 = self.sab1(d1)
        d1 = self.mscb1(d1)

        return [d4, d3, d2, d1]