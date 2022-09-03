
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def point_wise_maxpool(inputs, npts, keepdims=True):
    outputs = [torch.max(f, dim=2, keepdims=keepdims)[0] 
        for f in torch.split(inputs, npts, dim=2)]
    return torch.cat(outputs, axis=0)

def expand(global_features, npts):
    s = [1]*global_features.shape[0]
    inputs = torch.split(global_features, s, dim=0)
    outputs = [f.expand(f.shape[0],f.shape[1],npts[i]) for i,f in enumerate(inputs)]
    return torch.cat(outputs, axis=2)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # first shared mlp
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # second shared mlp
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
    
    def forward(self, x, npts):
        n = x.size()[2]

        # first shared mlp
        x = F.relu(self.bn1(self.conv1(x)))           # (1, 128, N)
        f = self.bn2(self.conv2(x))                   # (1, 256, N)

        # point-wise maxpool
        g = point_wise_maxpool(f, npts, keepdims=True) # (B, 256, 1)
        
        # expand and concat
        gs = expand(g, npts)
        x = torch.cat([gs, f], dim=1)                  # (B, 512, N)

        # second shared mlp
        x = F.relu(self.bn3(self.conv3(x)))           # (B, 512, N)
        x = self.bn4(self.conv4(x))                   # (B, 1024, N)
        
        # point-wise maxpool
        v = point_wise_maxpool(x, npts, keepdims=False) # (B, 1024)
        
        return v


class Decoder(nn.Module):
    def __init__(self, num_coarse=1024, num_dense=16384):
        super(Decoder, self).__init__()

        self.num_coarse = num_coarse
        
        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

        # shared mlp
        self.conv1 = nn.Conv1d(3+2+1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)

        # 2D grid
        grids = np.meshgrid(np.linspace(-0.05, 0.05, 4, dtype=np.float32),
                            np.linspace(-0.05, 0.05, 4, dtype=np.float32))                               # (2, 4, 44)
        self.grids = torch.Tensor(grids).view(2, -1)  # (2, 4, 4) -> (2, 16)
    
    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x  # (B, 1024)

        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 1024)

        v1 = y_coarse.unsqueeze(3)
        repeated_centers = v1.expand(v1.shape[0], v1.shape[1], v1.shape[2], 16).reshape(b, 3, -1)  # (B, 3, 16x1024)
        
        v2 = v.unsqueeze(2)
        repeated_v = v2.expand(v2.shape[0], v2.shape[1], 16 * self.num_coarse)               # (B, 1024, 16x1024)
        
        v3 = self.grids.to(x.device)  # (2, 16)
        v3 = v3.unsqueeze(0)
        grids = v3.repeat(b, 1, self.num_coarse)                      # (B, 2, 16x1024)

        x = torch.cat([repeated_v, grids, repeated_centers], dim=1)                  # (B, 2+3+1024, 16x1024)
        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.bn4(self.conv2(x)))
        x = self.conv3(x)                # (B, 3, 16x1024)
        y_detail = x + repeated_centers  # (B, 3, 16x1024)

        return y_coarse, y_detail


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x, npts):
        v = self.encoder(x, npts)
        y_coarse, y_detail = self.decoder(v)
        return v, y_coarse, y_detail
        
