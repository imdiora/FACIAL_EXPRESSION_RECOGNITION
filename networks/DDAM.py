from torch import nn
import torch
from networks import MixedFeatureNet
from torch.nn import Module
import os
from collections import OrderedDict

# Import necessary classes
from networks.MixedFeatureNet import MixedFeatureNet

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys if model was saved with DataParallel."""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()}

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding='same', groups=1):
        super(Linear_block, self).__init__()
        if padding == 'same':
            padding = (
                (kernel[0] - 1) // 2,
                (kernel[1] - 1) // 2
            )
        self.conv = nn.Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
class DDAMNet(nn.Module):
    def __init__(self, pretrained=True, num_head=3, num_class=7):
        super(DDAMNet, self).__init__()
        self.num_class = num_class
        self.num_heads = num_head

        # Initialize the network model
        net = MixedFeatureNet(num_classes=num_class, num_heads=num_head)

        if pretrained:
            # Load the pretrained model
            pretrained_model_path = os.path.join('./pretrained/', "MFN_msceleb.pth")
            if os.path.isfile(pretrained_model_path):
                pretrained_model = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

                # Check if the loaded object is a state_dict or a model object
                if isinstance(pretrained_model, dict):
                    state_dict = pretrained_model
                else:
                    state_dict = pretrained_model.state_dict()

                # Remove 'module.' prefix if present
                state_dict = remove_module_prefix(state_dict)

                # Load the state_dict
                net.load_state_dict(state_dict, strict=False)
            else:
                print(f"Pretrained model not found at {pretrained_model_path}")

        self.net = net

    # Define the forward pass
    def forward(self, x):
        return self.net(x)

        # Process through each `cat_head`
        for i in range(self.num_head):
                    heads.append(getattr(self, f"cat_head{i}")(x))


        y = heads[0]
        for i in range(1, self.num_head):
            y = torch.max(y, heads[i])
        

        y = x * y
        y = self.Linear(y)
    
        y = nn.AdaptiveAvgPool2d((1, 1))(y)
        y = torch.flatten(y, 1)
        out = self.fc(y)
        return out, x, heads
        
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
                      
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordAtt(512, 512)

    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()

        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding='same')
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding='same')

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(7, 7), stride=(1, 1), padding='same')
        self.flatten = Flatten()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.Linear_h(x)
        x_w = self.Linear_w(x)
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()

        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = x_w * x_h
        return y

