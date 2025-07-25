import torch
from torch import nn
import torch.nn.functional as F
from modules.netvlad import NetVLADLoupe
from torchvision.models.resnet import resnet18
import math
import copy

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Helper Modules from Original LCPR & MatViT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def paronamic_concat(x, N):
    if x.ndim == 4:
        BN, C, H, W = x.shape
        B = int(BN / N)
        x = x.view(B, N, C, H, W).permute(0, 2, 3, 1, 4).reshape(B, C, H, N * W)
    elif x.ndim == 3:
        BN, C, L = x.shape
        B = int(BN / N)
        x = x.view(B, N, C, L).permute(0, 2, 1, 3).reshape(B, C, N * L)
    return x

def paronamic_concat_inv(x, N):
    if x.ndim == 4:
        B, C, H, NW = x.shape
        W = int(NW / N)
        x = x.view(B, C, H, N, W).permute(0, 3, 1, 2, 4).reshape(B * N, C, H, W)
    elif x.ndim == 3:
        B, C, NL = x.shape
        L = int(NL / N)
        x = x.view(B, C, N, L).permute(0, 2, 1, 3).reshape(B * N, C, L)
    return x

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
        self.current_scale = 1.0
        return out

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
        self.current_mid_channels = self.mid_channels
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
        if x_i.ndim == 4:
            _, _, Hi, Wi = x_i.shape
            x_i = paronamic_concat(x_i, self.num_cam)
        else:
            Hi, Wi = 1, 1
        
        if x_l.ndim == 4:
            _, _, Hl, Wl = x_l.shape
        else:
            Hl, Wl = 1, 1

        x_i = self.v_conv_i(x_i)
        x_l = self.v_conv_l(x_l)
        x = torch.cat((x_i, x_l), dim=-1)
        x = x.transpose(1, 2)
        x = x + self.atten(x, x, x)
        x = x.transpose(1, 2)
        x_i, x_l = torch.split(x, [self.num_cam * Wi, Wl], dim=-1)

        if x_i.ndim == 3:
            x_i = x_i.unsqueeze(2).expand(-1, -1, Hi, -1)
            x_i = paronamic_concat_inv(x_i, self.num_cam)
        
        if x_l.ndim == 3:
            x_l = x_l.unsqueeze(2).expand(-1, -1, Hl, -1)

        return x_i, x_l

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Matformer-style Vision Transformer (MatViT)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#This Mlp class is not using any Matformer feature of elasticity, it must be modified.
class Mlp(nn.Module):
    """ Mlp with Matryoshka-style width scaling. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # Matryoshka attributes
        self.hidden_features_max = self.fc1.out_features
        self.width_scale = 1.0

    def configure_width(self, width_scale):
        self.width_scale = width_scale

    def forward(self, x):
        # Determine current hidden dimensions based on scale
        current_hidden_features = int(self.hidden_features_max * self.width_scale)
        
        # If scaled dimension is zero, return zero tensor to maintain structure
        if current_hidden_features == 0:
            return torch.zeros_like(self.fc2.bias)

        # --- Sliced Forward Pass ---
        # FC1: Slice output dimension
        x = F.linear(x, self.fc1.weight[:current_hidden_features, :], self.fc1.bias[:current_hidden_features])
        x = self.act(x)
        x = self.drop(x)
        # FC2: Slice input dimension
        x = F.linear(x, self.fc2.weight[:, :current_hidden_features], self.fc2.bias)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """ Attention with Matryoshka-style width scaling. """
    def __init__(self, dim, width_scale, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Matryoshka attributes
        self.width_scale = 1.0

    def configure_width(self, width_scale):
        self.width_scale = width_scale

    def forward(self, x):
        B, N, C = x.shape
        # Determine current dimensions based on scale
        current_head_dim = int(self.head_dim * self.width_scale)
        current_dim = current_head_dim * self.num_heads

        if current_dim == 0:
            return torch.zeros_like(x)

        # --- Sliced Forward Pass ---
        # QKV: Slice output dimension
        current_qkv_dim = current_dim * 3
        qkv = F.linear(x, self.qkv.weight[:current_qkv_dim, :], self.qkv.bias[:current_qkv_dim])
        qkv = qkv.reshape(B, N, 3, self.num_heads, current_head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.scale / (self.width_scale**0.5)) # Adjust scale for smaller dims

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, current_dim)
        
        # Proj: Slice input dimension
        x = F.linear(x_attn, self.proj.weight[:, :current_dim], self.proj.bias)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.width_scale = 1.0
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, self.width_scale, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def configure_block(self, width_scale):
        self.width_scale = width_scale
        self.attn.configure_width(width_scale)
        self.mlp.configure_width(width_scale)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MatViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192, depth=12,
                 num_heads=3, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.current_depth = depth
        

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.head = nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def configure_subnetwork(self, depth_scale, width_scale=1.0):
        self.current_depth = int(self.depth * depth_scale)
        for i in range(self.current_depth):
            self.blocks[i].configure_block(width_scale)
        

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i in range(self.current_depth):
            x = self.blocks[i](x)
        
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)

class Reshaper(nn.Module):
    def __init__(self, patch_embed):
        super().__init__()
        self.grid_size = patch_embed.grid_size
        self.embed_dim = patch_embed.proj.out_channels

    def forward(self, x):
        B, N, C = x.shape
        # Remove CLS token and reshape
        x = x[:, 1:, :].reshape(B, self.grid_size[0], self.grid_size[1], self.embed_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# The Main LCPR-MatViT Model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class LCPR_MatViT(nn.Module):
    def __init__(self, matvit_kwargs, image_backbone, lidar_backbone, fusion_blocks, final_v_convs, netvlads, eca_layer):
        super().__init__()
        # Image Encoders
        self.image_encoders = nn.ModuleList([MatViT(**matvit_kwargs) for _ in range(6)])
        self.reshapers = nn.ModuleList([Reshaper(enc.patch_embed) for enc in self.image_encoders])
        self.projection_i = image_backbone['projection']
        self.layer2_i = image_backbone['layer2_i']
        self.layer3_i = image_backbone['layer3_i']
        self.layer4_i = image_backbone['layer4_i']

        # LiDAR Encoder
        self.conv_l = lidar_backbone['conv_l']
        self.layer2_l = lidar_backbone['layer2_l']
        self.layer3_l = lidar_backbone['layer3_l']
        self.layer4_l = lidar_backbone['layer4_l']

        # Fusion Blocks
        self.fusion_blocks = fusion_blocks

        # Final Layers
        self.v_conv_i = final_v_convs['v_conv_i']
        self.v_conv_l = final_v_convs['v_conv_l']
        self.netvlad_i = netvlads['netvlad_i']
        self.netvlad_l = netvlads['netvlad_l']
        self.eca = eca_layer

    def configure_subnetwork(self, granularities):
        # Configure MatViT depth and width scaling
        for encoder in self.image_encoders:
            encoder.configure_subnetwork(granularities['vit_depth_scale'], granularities['vit_width_scale'])
        # Configure fusion block elasticity
        for i, block in enumerate(self.fusion_blocks):
            block.configure_subnetwork(granularities['fusion_granularities'][i])

    def forward(self, x_i, x_l):
        B, N, C, Hi, Wi = x_i.shape
        print(x_i.shape)
        # 1. Process each image with its MatViT encoder and reshape
        image_features_list = []
        for i in range(N):
            img = x_i[:, i, :, :, :]
            img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            features_seq = self.image_encoders[i](img_resized)
            print(features_seq.shape)
            features_map = self.reshapers[i](features_seq)
            print(f'Image feature maps {features_map.shape}')
            image_features_list.append(features_map)
        
        x_i = torch.cat(image_features_list, dim=0)
        print(x_i.shape)
        x_i = self.projection_i(x_i)
        print(f'Projected image features {x_i.shape}')
        print(f'LiDAR input shape {x_l.shape}')
        # 3. Process LiDAR and fuse at multiple scales
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
        
        # 4. Panoramic Concat and Final Layers
        x_i = paronamic_concat(x_i, N)
        x_i = self.v_conv_i(x_i)
        x_l = self.v_conv_l(x_l)

        x_i = self.netvlad_i(x_i.transpose(1, 2))
        x_l = self.netvlad_l(x_l.transpose(1, 2))
        x = torch.cat((x_i, x_l), dim=-1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.eca(x)
        x = x.squeeze(-1).squeeze(-1)
        x = nn.functional.normalize(x, dim=-1)

        return x

    @classmethod
    def create(cls, weights=None):
        # Matformer-style ViT parameters
        matvit_kwargs = {'img_size': 224, 'patch_size': 16, 'embed_dim': 192, 'depth': 4, 'num_heads': 4}
        
        # Original LCPR components
        pretrained = resnet18(weights=weights) if weights is not None else resnet18()
        
        # Image Backbone (ResNet layers for fusion)
        image_backbone = {
            'projection': nn.Conv2d(192, 64, kernel_size=1),
            'layer2_i': pretrained.layer2,
            'layer3_i': pretrained.layer3,
            'layer4_i': pretrained.layer4,
        }

        # LiDAR Backbone
        lidar_backbone = {
            'conv_l': nn.Conv2d(1, 64, kernel_size=1),
            'layer2_l': copy.deepcopy(pretrained.layer2),
            'layer3_l': copy.deepcopy(pretrained.layer3),
            'layer4_l': copy.deepcopy(pretrained.layer4),
        }

        # Fusion Blocks
        fusion_blocks = nn.ModuleList([
            fusion_block(in_channels=64, mid_channels=32, out_channels=64, num_cam=6),
            fusion_block(in_channels=128, mid_channels=64, out_channels=128, num_cam=6),
            fusion_block(in_channels=256, mid_channels=128, out_channels=256, num_cam=6),
            fusion_block(in_channels=512, mid_channels=256, out_channels=512, num_cam=6)
        ])

        # Final Layers
        final_v_convs = {
            'v_conv_i': VertConv(in_channels=512, mid_channels=256, out_channels=256),
            'v_conv_l': VertConv(in_channels=512, mid_channels=256, out_channels=256)
        }
        netvlads = {
            'netvlad_i': NetVLADLoupe(feature_size=256, max_samples=12, cluster_size=32, output_dim=128),
            'netvlad_l': NetVLADLoupe(feature_size=256, max_samples=132, cluster_size=32, output_dim=128)
        }
        eca = eca_layer(256)

        return cls(matvit_kwargs, image_backbone, lidar_backbone, fusion_blocks, final_v_convs, netvlads, eca)