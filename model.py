import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
import numpy as np

from config import cfg

def add_conv_block(in_ch=1, out_ch=1, filter_size=cfg.FIL_SIZE, dilate=1, last=False):
        conv_1 = nn.Conv2d(in_ch, out_ch, tuple(filter_size), padding=0, dilation=dilate)
        bn_1 = nn.BatchNorm2d(out_ch)

        return [conv_1, bn_1]

class MSDNet(nn.Module):
        """
        Paper: A mixed-scale dense convolutional neural network for image analysis
        Published: PNAS, Jan. 2018 
        Paper: http://www.pnas.org/content/early/2017/12/21/1715832114
        """
        @staticmethod
        def weight_init(m):
                if isinstance(m, nn.Linear):
                        torch.nn.init.kaiming_normal(m, m.weight.data)

        def __init__(self, num_layers=cfg.N_LAYERS, in_channels=None, out_channels=None):
                if in_channels is None:
                        in_channels=cfg.IN_CHANNELS

                if out_channels is None:
                        out_channels=cfg.N_CLASSES

                super(MSDNet, self).__init__()

                self.layer_list = add_conv_block(in_ch=in_channels)

                current_in_channels = 1
                # Add N layers
                for i in range(num_layers):
                        s1 = (i)%(cfg.DIL_M[0]) + 1
                        s2 = (i)%(cfg.DIL_M[1]) + 1
                        self.layer_list += add_conv_block(in_ch=current_in_channels, dilate=(s1,s2))
                        current_in_channels += 1

                # Add final output block
                self.layer_list += add_conv_block(in_ch=current_in_channels + in_channels, out_ch=out_channels, filter_size=(1,1), last=True)

                # Add to Module List
                self.layers = nn.ModuleList(self.layer_list)


        def forward(self, x):
                prev_features = []
                inp = x
                fil1 = (cfg.FIL_SIZE[0]-1)/2
                fil2 = (cfg.FIL_SIZE[1]-1)/2

                for i, f in enumerate(self.layers):
                        # Periodic & Zero paddings in x & t
                        # Check if last conv block
                        if i == (len(self.layers) - 2):
                                x = torch.cat(prev_features + [inp], 1)
                        elif (i)%2 == 0:
                                if i > 1:
                                    ilayer = i/2-1
                                    s1 = int(fil1 * (ilayer%cfg.DIL_M[0] + 1))
                                    s2 = int(fil2 * (ilayer%cfg.DIL_M[1] + 1))
                                    x_pad = F.pad(x, (0,0, s1,s1), "circular")
                                    x = F.pad(x_pad, (s2,s2, 0,0), "replicate") 
                                elif i == 0:
                                    s1 = int(fil1)
                                    s2 = int(fil2)
                                    x_pad = F.pad(x, (0,0,s1,s1), "circular")
                                    x = F.pad(x_pad, (s2,s2, 0,0), "replicate")



                        if (i+1)%2 == 0 and (not i==(len(self.layers)-1)):
                                x = F.relu(x)
                                # Append output into previous features
                                prev_features.append(x)
                                x = torch.cat(prev_features, 1)

                return x
