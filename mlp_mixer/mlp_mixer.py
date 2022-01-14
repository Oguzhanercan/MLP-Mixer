import torch
from torch import nn
import numpy as np
import einops
import torchvision
import torchvision.transforms as transforms

class Project(nn.Module):
    def __init__(self,N:int,in_channels,hidden_dim):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.linear1 = nn.Linear(self.in_channels*self.N**2,self.hidden_dim)
    def forward(self,x:torch.Tensor):
        out = einops.rearrange(x,"b c (h px) (w py) ->b (h w) (c px py)",px =self.N,py =self.N)
        out = self.linear1(out)
        return out
        
class token_mix(nn.Module):
    def __init__(self,in_dim,hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(self.in_dim,self.hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.hidden_dim,self.in_dim)
        self.layer_norm = nn.LayerNorm(self.in_dim)
    def forward(self,x:torch.Tensor):
        out = einops.rearrange(x,"b f s -> b s f")
        out = self.layer_norm(out)
        out = self.linear1(out)
        out = self.gelu(out)
        out = self.linear2(out)
        out = einops.rearrange(out,"b f s -> b s f")
        return out

class channel_mix(nn.Module):
    def __init__(self,in_dim,hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(self.in_dim,self.hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.hidden_dim,self.in_dim)
        self.layer_norm = nn.LayerNorm(self.in_dim)
    def forward(self,x:torch.Tensor):
        out = self.layer_norm(x)
        out = self.linear1(out)
        out = self.gelu(out)
        out = self.linear2(out)
        return out
        
        
class MixerLayer(nn.Module):
    def __init__(self,num_patches,token_mix_hidden,hidden_dim,channel_mix_hidden):
        super().__init__()
        self.token = token_mix(num_patches,token_mix_hidden)
        self.channel = channel_mix(hidden_dim,channel_mix_hidden)

    def forward(self,x:torch.Tensor):
        out= x + self.token(x)
        out = out + self.channel(out)
        return out
        
        
class Classifier(nn.Module):
    def __init__(self,in_dim,num_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_dim,num_classes)
    def forward(self,x:torch.Tensor):
        return self.linear1(x)
        
        
class Mixer(nn.Module):
    def __init__(self,num_classes,in_channels = 3,img_size = 224,patch_size = 8,hidden_dim = 512,token_mix_hidden = 256,channel_mix_hidden = 2048,num_layer =8):
        super().__init__()
        self.num_patches = (img_size//patch_size)**2
        self.projector = Project(patch_size,in_channels,hidden_dim)
        self.layers = [MixerLayer(self.num_patches,token_mix_hidden,hidden_dim,channel_mix_hidden) for _ in range(num_layer)]
        self.classifier = Classifier(hidden_dim,num_classes)
        self.MlpMixer = nn.Sequential(*self.layers)
    def forward(self,x :torch.Tensor):
        out = self.projector(x)
        out = self.MlpMixer(out)
        out = torch.mean(out,dim = 1)
        out = self.classifier(out)
        return out
        
        
        
                        
        
        
        
        
        

