import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18

class RidgeDetector(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        layers = vgg16(pretrained=model_args['pretrained']).features
        input_format = model_args['input_format']
        self.model_args = model_args

        self.zeroth_layer = nn.Identity()

        # RGB + Depth as array (C:4) -> 1x1 conv -> (C:3)
        if input_format == "rgb-darr":
            self.zeroth_layer = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1, padding=0)
        
        # RGB + Depth as image (C:6) -> 1x1 conv -> (C:3)
        elif input_format == "rgb-drgb":
            self.zeroth_layer = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Depth as array only (C:1)
        elif input_format == "darr":
            # TODO: Currently using broadcasted input, several alternatives available:
            # Reduce weights of existing conv layer (avg)
            # Insert a new conv layer with proper channels 
            pass

        # RGB image only (C:3) or Depth as image only (C:3)
        elif input_format in ("rgb", "drgb"):
            pass # Use the network as it is

        else:
            raise NotImplementedError

        # Stage-1: Layers 0-3
        self.stage1 = layers[0:4]
        self.sideout1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0), # 1x1 conv to shrink channels
            nn.Upsample(scale_factor=1, mode="bilinear", align_corners=False)
        )
        # Stage-2: Layers 4-8
        self.stage2 = layers[4:9]
        self.sideout2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0), # 1x1 conv to shrink channels
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        # Stage-3: Layers 9-15
        self.stage3 = layers[9:16]
        self.sideout3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0), # 1x1 conv to shrink channels
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        )
        # Stage-4: Layers 16-22
        self.stage4 = layers[16:23]
        self.sideout4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0), # 1x1 conv to shrink channels
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        )
        # Stage-5: Layers 23-29
        self.stage5 = layers[23:30]
        self.sideout5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0), # 1x1 conv to shrink channels
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)
        )
        # Fuse side outputs from 5 stages at the end
        self.fuse = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0)

    def get_args(self):
        return self.model_args

    def forward(self, x):
        # x: NxCxHxW -> y: NxCxHxW -> sideouts: 5xNxCxHxW
        x = self.zeroth_layer(x)

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

        y = torch.cat([y1, y2, y3, y4, y5], dim=1) # Merge side outputs in a tensor
        y6 = self.fuse(y)                          # Fuse side outputs
        y = torch.cat([y, y6], dim=1)              # Merge of all outputs in a tensor
        
        return y

def main():
    rd = RidgeDetector()
    print(rd)

if __name__ == "__main__":
    main()