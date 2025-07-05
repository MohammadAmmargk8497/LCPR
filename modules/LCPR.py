import math
import copy
import torch
from torch import nn
import torch.nn.functional as F
from modules.netvlad import NetVLADLoupe
from torchvision.models.resnet import resnet18


class LCPR(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.conv1_i = pretrained.conv1
        self.bn1_i = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.conv_l = nn.Conv2d(1, 64, kernel_size=1)
        self.layer1_i = pretrained.layer1
        self.layer2_i = pretrained.layer2
        self.layer2_l = copy.deepcopy(pretrained.layer2)
        self.layer3_i = pretrained.layer3
        self.layer3_l = copy.deepcopy(pretrained.layer3)
        self.layer4_i = pretrained.layer4
        self.layer4_l = copy.deepcopy(pretrained.layer4)
        self.fusion_blocks = nn.ModuleList([
            fusion_block(in_channels=64, mid_channels=32, out_channels=64, num_cam=6),
            fusion_block(in_channels=128, mid_channels=64, out_channels=128, num_cam=6),
            fusion_block(in_channels=256, mid_channels=128, out_channels=256, num_cam=6),
            fusion_block(in_channels=512, mid_channels=256, out_channels=512, num_cam=6)
        ])
        self.v_conv_i = VertConv(in_channels=512, mid_channels=256, out_channels=256)
        self.v_conv_l = VertConv(in_channels=512, mid_channels=256, out_channels=256)
        self.netvlad_i = NetVLADLoupe(feature_size=256, max_samples=132, cluster_size=32, output_dim=128)
        self.netvlad_l = NetVLADLoupe(feature_size=256, max_samples=132, cluster_size=32, output_dim=128)
        self.eca = eca_layer(256)

    def configure_subnetwork(self, granularities):
        for i, block in enumerate(self.fusion_blocks):
            block.configure_subnetwork(granularities[i])

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        non_elastic_params = total_params
        for block in self.fusion_blocks:
            non_elastic_params -= block.atten.get_num_params()
            non_elastic_params -= block.v_conv_i.get_num_params()
            non_elastic_params -= block.v_conv_l.get_num_params()

        elastic_params = 0
        for block in self.fusion_blocks:
            elastic_params += block.atten.get_num_params_scaled()
            elastic_params += block.v_conv_i.get_num_params_scaled()
            elastic_params += block.v_conv_l.get_num_params_scaled()

        return non_elastic_params + elastic_params

    def forward(self, x_i, x_l):
        B, N, C, Hi, Wi = x_i.shape
        x_i = x_i.view(B * N, C, Hi, Wi)
        x_i = self.conv1_i(x_i)
        x_i = self.bn1_i(x_i)
        x_i = self.relu(x_i)
        x_i = self.maxpool(x_i)
        x_i = self.layer1_i(x_i)

        x_l = self.conv_l(x_l)
        x_i_1, x_l_1 = self.fusion_blocks[0](x_i, x_l)
        x_i = x_i + x_i_1
        x_l = x_l + x_l_1

        x_i = self.layer2_i(x_i)
        x_l = self.layer2_l(x_l)
        x_i_1, x_l_1 = self.fusion_blocks[1](x_i, x_l)
        x_i = x_i + x_i_1
        x_l = x_l + x_l_1

        x_i = self.layer3_i(x_i)
        x_l = self.layer3_l(x_l)
        x_i_1, x_l_1 = self.fusion_blocks[2](x_i, x_l)
        x_i = x_i + x_i_1
        x_l = x_l + x_l_1

        x_i = self.layer4_i(x_i)
        x_l = self.layer4_l(x_l)
        x_i_1, x_l_1 = self.fusion_blocks[3](x_i, x_l)
        x_i = x_i + x_i_1
        x_l = x_l + x_l_1

        x_i = paronamic_concat(x_i, N=6)
        x_i = self.v_conv_i(x_i)
        x_i = x_i.unsqueeze(2)
        x_l = self.v_conv_l(x_l)
        x_l = x_l.unsqueeze(2)

        x_i = self.netvlad_i(x_i)
        x_l = self.netvlad_l(x_l)
        x = torch.cat((x_i, x_l), dim=-1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.eca(x)
        x = x.squeeze(-1).squeeze(-1)
        x = nn.functional.normalize(x, dim=-1)

        return x

    @classmethod
    def create(cls, weights=None):
        pretrained = resnet18(weights=weights) if weights is not None else resnet18()
        return cls(pretrained)


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def paronamic_concat(x, N):
    BN, C, H, W = x.shape
    B = int(BN / N)
    x = x.view(B, N, C, H, W).permute(0, 2, 3, 1, 4).reshape(B, C, H, N * W)
    return x


def paronamic_concat_inv(x, N):
    B, C, H, NW = x.shape
    W = int(NW / N)
    x = x.view(B, C, H, N, W).permute(0, 3, 1, 2, 4).reshape(B * N, C, H, W)
    return x


class fusion_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_cam):
        super().__init__()
        self.num_cam = num_cam
        self.v_conv_i = VertConv(in_channels, mid_channels, out_channels)
        self.v_conv_l = VertConv(in_channels, mid_channels, out_channels)
        self.atten = MultiHeadAttention(d_model=out_channels, n_head=4)

    def configure_subnetwork(self, granularity):
        self.atten.configure_subnetwork(granularity['scale'])
        self.v_conv_i.configure_subnetwork(granularity['mid_channel_scale'])
        self.v_conv_l.configure_subnetwork(granularity['mid_channel_scale'])

    def forward(self, x_i, x_l):
        _, _, Hi, Wi = x_i.shape
        _, _, Hl, Wl = x_l.shape
        x_i = paronamic_concat(x_i, self.num_cam)
        x_i = self.v_conv_i(x_i)
        x_l = self.v_conv_l(x_l)
        x = torch.cat((x_i, x_l), dim=-1)
        x = x.transpose(1, 2)
        x = x + self.atten(x, x, x)
        x = x.transpose(1, 2)
        x_i, x_l = torch.split(x, [self.num_cam * Wi, Wl], dim=-1)

        x_i = x_i.unsqueeze(2).expand(-1, -1, Hi, -1)
        x_i = paronamic_concat_inv(x_i, self.num_cam)

        x_l = x_l.unsqueeze(2).expand(-1, -1, Hl, -1)
        return x_i, x_l


class VertConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.current_mid_channels = mid_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        )
        self.reduce_conv = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm1d(mid_channels)
        self.relu_reduce = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.out_conv = nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.bn_out = nn.BatchNorm1d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)

    def configure_subnetwork(self, scale):
        self.current_mid_channels = int(self.mid_channels * scale)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params_scaled(self):
        reduce_params = self.current_mid_channels * self.in_channels * 1
        bn_reduce_params = 2 * self.current_mid_channels
        conv1_params = self.current_mid_channels * self.current_mid_channels * 3
        bn1_params = 2 * self.current_mid_channels
        conv2_params = self.current_mid_channels * self.current_mid_channels * 3
        bn2_params = 2 * self.current_mid_channels
        out_conv_params = self.out_channels * self.current_mid_channels * 1 + self.out_channels
        bn_out_params = 2 * self.out_channels
        return reduce_params + bn_reduce_params + conv1_params + bn1_params + conv2_params + bn2_params + out_conv_params + bn_out_params

    def forward(self, x):
        x = self.input_conv(x).max(2)[0] 
        
        x = F.conv1d(x, self.reduce_conv.weight[:self.current_mid_channels, :, :])
        x = F.batch_norm(x, self.bn_reduce.running_mean[:self.current_mid_channels], self.bn_reduce.running_var[:self.current_mid_channels], self.bn_reduce.weight[:self.current_mid_channels], self.bn_reduce.bias[:self.current_mid_channels], training=self.bn_reduce.training)
        x = self.relu_reduce(x)
        
        identity = x

        x = F.conv1d(x, self.conv1.weight[:self.current_mid_channels, :self.current_mid_channels, :], padding=1)
        x = F.batch_norm(x, self.bn1.running_mean[:self.current_mid_channels], self.bn1.running_var[:self.current_mid_channels], self.bn1.weight[:self.current_mid_channels], self.bn1.bias[:self.current_mid_channels], training=self.bn1.training)
        x = self.relu1(x)

        x = F.conv1d(x, self.conv2.weight[:self.current_mid_channels, :self.current_mid_channels, :], padding=1)
        x = F.batch_norm(x, self.bn2.running_mean[:self.current_mid_channels], self.bn2.running_var[:self.current_mid_channels], self.bn2.weight[:self.current_mid_channels], self.bn2.bias[:self.current_mid_channels], training=self.bn2.training)
        x = self.relu2(x)

        x = x + identity

        x = F.conv1d(x, self.out_conv.weight[:, :self.current_mid_channels, :], self.out_conv.bias)
        x = self.bn_out(x)
        x = self.relu_out(x)

        self.current_mid_channels = self.mid_channels # Reset
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.attention_head_size = d_model // n_head
        self.current_scale = 1.0
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def configure_subnetwork(self, scale):
        self.current_scale = scale

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params_scaled(self):
        current_head_size = int(self.attention_head_size * self.current_scale)
        all_head_size = self.n_head * current_head_size

        q_params = all_head_size * self.d_model + all_head_size
        k_params = all_head_size * self.d_model + all_head_size
        v_params = all_head_size * self.d_model + all_head_size
        out_params = self.d_model * all_head_size + self.d_model

        return q_params + k_params + v_params + out_params

    def transpose_for_scores(self, x, head_size):
        new_x_shape = x.size()[:-1] + (self.n_head, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        current_head_size = int(self.attention_head_size * self.current_scale)
        if current_head_size == 0: return torch.zeros_like(q)

        all_head_size = self.n_head * current_head_size

        q_s = F.linear(q, self.w_q.weight[:all_head_size, :], self.w_q.bias[:all_head_size])
        k_s = F.linear(k, self.w_k.weight[:all_head_size, :], self.w_k.bias[:all_head_size])
        v_s = F.linear(v, self.w_v.weight[:all_head_size, :], self.w_v.bias[:all_head_size])

        q_s = self.transpose_for_scores(q_s, current_head_size)
        k_s = self.transpose_for_scores(k_s, current_head_size)
        v_s = self.transpose_for_scores(v_s, current_head_size)

        out, _ = self.attention(q_s, k_s, v_s, mask=mask)
        out = out.permute(0, 2, 1, 3).contiguous()
        out_shape = out.size()[:-2] + (all_head_size,)
        out = out.view(*out_shape)
        
        out = F.linear(out, self.w_concat.weight[:, :all_head_size], self.w_concat.bias)

        self.current_scale = 1.0  # Reset after forward pass
        return out


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__() 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        d_tensor = k.size(-1)
        k_t = k.transpose(-2, -1)
        score = (q @ k_t) / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score

