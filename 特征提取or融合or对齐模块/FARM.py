import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import DeformConv2d
from pytorch_lightning import seed_everything
from einops import rearrange
import numbers

seed_everything(13)


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(pl.LightningModule):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(pl.LightningModule):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(pl.LightningModule):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(pl.LightningModule):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(pl.LightningModule):
    def __init__(self, dim, num_heads, stride, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.stride = stride
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=self.stride, padding=1, groups=dim * 2,
                                   bias=bias)

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qk = self.qk_dwconv(self.qk(x))
        q, k = qk.chunk(2, dim=1)

        v = self.v_dwconv(self.v(x))

        b, f, h1, w1 = q.size()

        q = rearrange(q, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
######### Burst Feature Attention ########################################

##########################################################################
class BFA(pl.LightningModule):
    def __init__(self, dim, num_heads, stride, ffn_expansion_factor, bias, LayerNorm_type):
        super(BFA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, stride, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(pl.LightningModule):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        # print("Inside patch embed:::", inp_enc_level1.size())
        x = self.proj(x)

        return x


class ref_back_projection(pl.LightningModule):
    def __init__(self, in_channels, stride):
        super(ref_back_projection, self).__init__()

        bias = False

        self.feat_fusion = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1), nn.GELU())
        self.feat_expand = nn.Sequential(nn.Conv2d(in_channels, in_channels * 2, 3, stride=1, padding=1), nn.GELU())
        self.diff_fusion = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1), nn.GELU())

        self.encoder1 = nn.Sequential(*[
            BFA(dim=in_channels * 2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias,
                LayerNorm_type='WithBias') for i in range(2)])

    def forward(self, x):
        B, f, H, W = x.size()

        ref = x[0].unsqueeze(0)
        ref = torch.repeat_interleave(ref, B, dim=0)
        feat = self.encoder1(torch.cat([ref, x], dim=1))

        fused_feat = self.feat_fusion(feat)
        exp_feat = self.feat_expand(fused_feat)

        residual = exp_feat - feat
        residual = self.diff_fusion(residual)

        fused_feat = fused_feat + residual

        return fused_feat


class alignment(pl.LightningModule):
    def __init__(self, dim=48, memory=False, stride=1, type='group_conv'):

        super(alignment, self).__init__()

        act = nn.GELU()
        bias = False

        kernel_size = 3
        padding = kernel_size // 2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size ** 2)

        self.offset_conv = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform = DeformConv2d(dim, dim, kernel_size, padding=2, groups=deform_groups, dilation=2)
        self.back_projection = ref_back_projection(dim, stride=1)

        self.bottleneck = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=bias), act)

        if memory == True:
            self.bottleneck_o = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=bias), act)

    def offset_gen(self, x):

        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return offset, mask

    def forward(self, x, prev_offset_feat=None):

        B, f, H, W = x.size()
        ref = x[0].unsqueeze(0)
        ref = torch.repeat_interleave(ref, B, dim=0)

        offset_feat = self.bottleneck(torch.cat([ref, x], dim=1))

        if not prev_offset_feat == None:
            offset_feat = self.bottleneck_o(torch.cat([prev_offset_feat, offset_feat], dim=1))

        offset, mask = self.offset_gen(self.offset_conv(offset_feat))

        aligned_feat = self.deform(x, offset, mask)
        aligned_feat[0] = x[0].unsqueeze(0)

        aligned_feat = self.back_projection(aligned_feat)

        # return aligned_feat, offset_feat
        return aligned_feat

