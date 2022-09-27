import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CoordAtt import CoordAtt
from einops import rearrange
from models.preprocess import AugmentMelSTFT


def patchout(x, patchout=0.5):
    dim = x.shape[2]
    random_idx = torch.randperm(dim)[:int(dim*(1-patchout))].sort().values
    x = x[:, :, random_idx] 
    return x 
    

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): 
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                CoordAtt(hidden_dim, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, patch_stride, patchout, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.patchout = patchout
        self.patch_stride = patch_stride

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()
        # Stracting patches from original input
        y = F.unfold(y, kernel_size=(self.ph,self.pw), stride=self.patch_stride, padding=0, dilation=1)
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # Global representations
        ## Stracting patches from global features
        x = F.unfold(x, kernel_size=(self.ph,self.pw), stride=self.patch_stride, padding=0, dilation=1)
        ## Doing Unstructured Patchout 
        if self.training and self.patchout:
                x = patchout(x, patchout=self.patchout)
                y = patchout(y, patchout=self.patchout)
        y = rearrange(y, 'b (c h) w -> b h c w', c=self.ph*self.pw)    
        x = rearrange(x, 'b (c h) w -> b c w h', c=self.ph*self.pw)

        x = self.transformer(x)

        x = x.permute(0,3,1,2)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViTCA(nn.Module):
    def __init__(self, spec_size, dims, channels, num_classes, expansion=4, kernel_size=3, 
                 patch_size=(2, 2), patch_stride=(1,1), patchout=0.5):
                 
        super().__init__()
        ih, iw = spec_size
        
        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(1, channels[0], stride=1)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, patch_stride, patchout, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, patch_stride, patchout, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, patch_stride, patchout, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)
        
        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def mobilevitca_xxs(spec_size=(None,None), num_classes=1000, patch_size=(None, None),
                    patch_stride=(None,None), patchout=0.5):
                    
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    
    return MobileViTCA(spec_size, dims, channels, num_classes, 2,3,
                        patch_size, patch_stride, patchout)


def mobilevitca_xs(spec_size=(None,None), num_classes=1000, 
                    patch_size=(None, None), patch_stride=(None,None), patchout=0.5):
                    
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    
    return MobileViTCA(spec_size, dims, channels, num_classes, 4,3,
                        patch_size, patch_stride, patchout) 


def mobilevitca_s(spec_size=(None,None), num_classes=1000, 
                    patch_size=(None, None), patch_stride=(None,None), patchout=0.5):
                    
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    
    return MobileViTCA(spec_size, dims, channels, num_classes, 4, 3,
                        patch_size, patch_stride, patchout)    
                        
 
# This function will create our model
class make_model(nn.Module):
    def __init__(self, mel = AugmentMelSTFT(), net = 'xs', input_shape=(128,256), num_classes=4,
                 patch_size=(4, 4), patch_stride=(3, 3), patchout=0.5):   
                 
        """
        @param mel: spectrogram extractor
        @param net: network module
        """   
        super().__init__()
        self.mel = mel
        
        if net =='xxs':
            net = mobilevitca_xxs(input_shape, num_classes, 
                                      patch_size, patch_stride, patchout)
        
        elif net == 'xs':
            net = mobilevitca_xs(input_shape, num_classes, 
                                      patch_size, patch_stride, patchout)
                                      
        elif net == 's':
            net = mobilevitca_s(input_shape, num_classes, 
                                      patch_size, patch_stride, patchout)
                                      
        else:
            raise NotImplementedError
            
        self.net = net
        
    def forward(self, x):
        spec = self.mel(x)
        spec = spec.unsqueeze(1)
        x = self.net(spec)
        
        return x