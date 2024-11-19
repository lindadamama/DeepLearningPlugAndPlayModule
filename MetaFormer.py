import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.layers import to_2tuple


class Downsampling(nn.Module):
    """ Downsampling implemented by a layer of convolution. """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x

class Scale(nn.Module):
    """ Scale vector by element multiplications. """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class LayerNormGeneral(nn.Module):
    """ General LayerNorm for different situations. """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True,
                 bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models. """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.ReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MetaFormerBlock(nn.Module):
    """ Implementation of one MetaFormer block. """
    def __init__(self, dim, token_mixer=nn.Identity, mlp=Mlp, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0., layer_scale_init_value=None, res_scale_init_value=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x) + self.layer_scale2(self.drop_path2(self.mlp(self.norm2(x))))
        return x

class MetaFormer(nn.Module):
    """ MetaFormer main model. """
    def __init__(self, in_chans=3, num_classes=1000, depths=[2, 2, 6, 2], dims=[64, 128, 320, 512],
                 token_mixers=nn.Identity, mlps=Mlp, norm_layers=nn.LayerNorm,
                 drop_path_rate=0., head_dropout=0.0, layer_scale_init_values=None,
                 res_scale_init_values=None, output_norm=nn.LayerNorm, head_fn=nn.Linear, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        num_stage = len(depths)
        self.num_stage = num_stage

        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList([Downsampling(down_dims[i], down_dims[i + 1], kernel_size=3, stride=2, padding=1)
                                                for i in range(num_stage)])

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(num_stage):
            stage = nn.Sequential(*[MetaFormerBlock(dim=dims[i], token_mixer=token_mixers,
                                                    mlp=mlps, norm_layer=norm_layers,
                                                    drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])
        self.head = head_fn(dims[-1], num_classes)

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
