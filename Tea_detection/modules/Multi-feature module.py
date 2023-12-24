import numpy as np
import torch
import torch.nn as nn

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p
class SiLU(nn.Module):                       # activate function
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):                       # convolution module
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

def conv_dw(filter_in, filter_out, stride = 1):          # Depth-separable convolution
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1    = nn.Linear(in_features, hidden_features)       # Mlp(160,256,512)
        self.dwconv = conv_dw(hidden_features, hidden_features)
        self.fc2    = nn.Linear(hidden_features, out_features)
        self.drop   = nn.Dropout(drop)
        self.c1     = hidden_features
        self.c2     = out_features

    def forward(self, x):
        #xin = x                                                        # (B,169,160)    

        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)                                          # (B,H,W,C)
        x = x.reshape(B,-1,C)                                           # (B,13*13,160) 
        x = self.fc1(x)                                                 # (B,169,160)->(B,169,256)
        xin = x.transpose(1,2).view(-1,self.c1,H,W)                     # (B,256,169)->(B,256,13,13)
        B,C,H,W = xin.shape                                             # B=B,C=256,H=13,W=13
        x = self.dwconv(xin)                                            # (B,256,13,13)
        x = self.drop(x)
        x = xin + x                                                     # (B,256,13,13)

        x = x.flatten(2).transpose(1, 2)                                # (B,256,169)->(B,169,256)
        x = self.fc2(x)                                                 # (B,169,512)
        x = self.drop(x)

        x = x.transpose(1,2).view(-1,self.c2,H,W)                       # (B,512,169)->(B,512,13,13)
        return x
    

class Multi_feature(nn.Module):
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(Multi_feature, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)

        self.Mlp = Mlp(c1,c1,c_)
        
        self.cv7 = Conv(3 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)                                                        # y1 and y2 are in local feature fusion
        y3 = self.Mlp(x)                                                        # y3 is in global feature fusion
        return self.cv7(torch.cat((y1, y2, y3), dim=1))
