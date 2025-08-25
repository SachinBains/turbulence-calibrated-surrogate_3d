import torch, torch.nn as nn
def convbn(in_ch, out_ch, dropout=0.0):
    layers = [nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout3d(p=dropout))
    else:
        layers.append(nn.Identity())
    return nn.Sequential(*layers)

class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32, dropout=0.0):
        super().__init__()
        self.dropout_p = dropout
        self.enc1 = nn.Sequential(convbn(in_ch, base_ch, dropout), convbn(base_ch, base_ch, dropout))
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(convbn(base_ch, base_ch*2, dropout), convbn(base_ch*2, base_ch*2, dropout))
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = nn.Sequential(convbn(base_ch*2, base_ch*4, dropout), convbn(base_ch*4, base_ch*4, dropout))
        self.pool3 = nn.MaxPool3d(2)
        self.bott = nn.Sequential(convbn(base_ch*4, base_ch*8, dropout), convbn(base_ch*8, base_ch*8, dropout))
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = nn.Sequential(convbn(base_ch*8, base_ch*4, dropout), convbn(base_ch*4, base_ch*4, dropout))
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = nn.Sequential(convbn(base_ch*4, base_ch*2, dropout), convbn(base_ch*2, base_ch*2, dropout))
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(convbn(base_ch*2, base_ch, dropout), convbn(base_ch, base_ch, dropout))
        self.out = nn.Conv3d(base_ch, out_ch, 1)
        self._mc_p = dropout
    def enable_mc_dropout(self, p=0.2):
        def setp(m):
            if isinstance(m, nn.Dropout3d): m.p = p
        self.apply(setp)
        self._mc_p = p
        self.train(True)
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bott(p3)
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], 1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)
        return self.out(d1)
