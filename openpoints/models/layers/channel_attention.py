from re import X
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import create_convblock1d, create_convblock2d

class BAFM(nn.Module):
    '''
    x:b, c, n
    '''
    def __init__(self, in_channels, out_channels=None, ration=4, norm_args=None, act_args=None):
        super(BAFM, self).__init__()
        mid_channels = in_channels * ration

        if out_channels==None:
            out_channels=in_channels
        
        
        self.fc1 = nn.Sequential(create_convblock1d(out_channels, mid_channels, norm_args=norm_args, act_args=act_args),
                              create_convblock1d(mid_channels, out_channels, norm_args=None, act_args=None),)
        
        self.fc2 = nn.Sequential(create_convblock1d(out_channels, mid_channels, norm_args=norm_args, act_args=act_args),
                              create_convblock1d(mid_channels, out_channels, norm_args=None, act_args=None),)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        '''
        x1:b, c1*2, n
        x2:b, c1, n
        '''
        
        y_max = torch.max(x2, -1, keepdim=True)[0]
        y_mean = torch.mean(x2, -1, keepdim=True)
        
        y_max = self.fc1(y_max)
        y_mean = self.fc2(y_mean)
        
        out = y_max*0.9 + y_mean*0.1
        out = self.sigmoid(out)
        out1 = out*x1 + x2
        out2 = out*x2 + x2
        x = torch.cat((out1, out2), dim=1)
        
        return x