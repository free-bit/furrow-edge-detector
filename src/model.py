import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18

class RidgeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        layers = vgg16(pretrained=True).features

        # Stage-1: Layers 0-3
        self.stage1 = layers[0:4]
        self.sideout1 = nn.Upsample(scale_factor=1, mode="bilinear", align_corners=True)
        
        # Stage-2: Layers 4-8
        self.stage2 = layers[4:9]
        self.sideout2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        # Stage-3: Layers 9-15
        self.stage3 = layers[9:16]
        self.sideout3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        
        # Stage-4: Layers 16-22
        self.stage4 = layers[16:23]
        self.sideout4 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        
        # Stage-5: Layers 23-29
        self.stage5 = layers[23:30]
        self.sideout5 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
    
    def forward(self, x):
        # x: NxCxHxW -> y: NxCxHxW -> sideouts: 5xNxCxHxW

        x = self.stage1(x)
        y1 = self.sideout1(x)

        x = self.stage2(x)
        y2 = self.sideout2(x)

        x = self.stage3(x)
        y3 = self.sideout3(x)

        x = self.stage4(x)
        y4 = self.sideout4(x)

        x = self.stage5(x)
        y5 = self.sideout5(x)
        
        return [y1, y2, y3, y4, y5]

def main():
    rd = RidgeDetector()
    print(rd)

if __name__ == "__main__":
    main()