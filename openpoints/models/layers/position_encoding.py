import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import create_convblock1d, create_convblock2d

class HPE(nn.Module):
    def __init__(self, in_channels, mode=1, norm_args=None, act_args=None):
        """A PosPool operator for local aggregation

        Args:
            in_channels: input channels.
        """
        super(HPE, self).__init__()
        self.in_channels = in_channels
        self.mode = mode
        
        if mode==1:
            self.conv =  nn.Sequential(create_convblock2d(3, in_channels//4, norm_args=norm_args, act_args=act_args), 
                                                                create_convblock2d(in_channels//4, in_channels, norm_args=None, act_args=None))
        elif mode==2:
            self.conv =  nn.Sequential(create_convblock2d(3, 3, norm_args=norm_args, act_args=act_args), 
                                                                create_convblock2d(3, in_channels, norm_args=None, act_args=None))
        elif mode==4:
            self.conv4 = create_convblock2d((in_channels//6)*6, in_channels, norm_args=norm_args, act_args=act_args)

    def forward(self, xyz):
        """
        Args:
            xyz: [B, 3, N, k]
            feature: [b, c, n, k]

        Returns:
           position: [B, C_out, 3]
        """
        B = xyz.shape[0]
        C = self.in_channels
        npoint = xyz.shape[2]
        k = xyz.shape[-1]

        if self.mode == 1:
           feature_new =  self.conv(xyz)
        elif self.mode==4:
            feat_dim = C // 6
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * xyz, -1)  # (B, 3, npoint, k, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, k, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, k, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, k, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, k, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.reshape(B, (C//6)*6, npoint, k).contiguous()  # (B, 3*C, npoint, k)
            position_embedding = self.conv4(position_embedding)
            #feature_new = feature * position_embedding  # (B, C, npoint)
            feature_new = position_embedding  # (B, C, npoint)

        return feature_new