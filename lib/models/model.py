# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

d = {'resnet18': {'models': resnet18, 'out': [32, 64, 128, 256, 512]},
     'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
     'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
     'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
     'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
     }

inplace = True


class CRNN_FPN(nn.Module):
    def  __init__(self, backbone, result_num=3, scale: int = 1, pretrained=False,onnx=False):
        super(CRNN_FPN, self).__init__()
        self.scale = scale
        conv_out = 512
        self.onnx = onnx
        model, out = d[backbone]['models'], d[backbone]['out']
        self.backbone = model(pretrained=pretrained)
        # Reduce channels
        # Top layer
        self.toplayer = nn.Sequential(nn.Conv2d(out[4], conv_out, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(conv_out),
                                      #nn.ReLU(inplace=inplace)
                                      )
        # Lateral layers
        self.latlayer1 = nn.Sequential(nn.Conv2d(out[3], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       #nn.ReLU(inplace=inplace)
                                       )
        self.latlayer2 = nn.Sequential(nn.Conv2d(out[2], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       #nn.ReLU(inplace=inplace)
                                       )
        self.latlayer3 = nn.Sequential(nn.Conv2d(out[1], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       #nn.ReLU(inplace=inplace)
                                       )
        self.latlayer4 = nn.Sequential(nn.Conv2d(out[0], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       #nn.ReLU(inplace=inplace)
                                       )

        self.smooth1 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
                                     #nn.BatchNorm2d(conv_out),
                                     #nn.ReLU(inplace=inplace),
                                     #nn.Conv2d(conv_out, conv_out, kernel_size=1, padding=0, stride=1),
                                     #nn.BatchNorm2d(conv_out),
                                     #nn.ReLU(inplace=inplace)
                                     )
        self.smooth2 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
                                     #nn.BatchNorm2d(conv_out),
                                     #nn.ReLU(inplace=inplace),
                                     #nn.Conv2d(conv_out, conv_out, kernel_size=1, padding=0, stride=1),
                                     #nn.BatchNorm2d(conv_out),
                                     #nn.ReLU(inplace=inplace)
                                     )
        self.smooth3 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
                                     #nn.BatchNorm2d(conv_out),
                                     #nn.ReLU(inplace=inplace),
                                     #nn.Conv2d(conv_out, conv_out, kernel_size=1, padding=0, stride=1),
                                     #nn.BatchNorm2d(conv_out),
                                     #nn.ReLU(inplace=inplace)
                                     )

        self.conv = nn.Sequential(
            nn.Conv2d(conv_out, conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_conv = nn.Conv2d(conv_out, result_num, kernel_size=1, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 0))
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 0))
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))


    def forward(self, c1, c2, c3, c4, c5):
        #_, _, H, W = input.size()
        #c2, c3, c4, c5 = self.backbone(input)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self.latlayer1(c4)
        #p4 = self._upsample_add_(p5, self.latlayer1(c4))
        #p4 = self.smooth1(p4)
        p3 = self.latlayer2(c3)
        #p3 = self.smooth2(p3)
        p2 = self.latlayer3(c2)
        #p2 = self.smooth3(p2)
        p1 = self.latlayer4(c1)
        #p1 = self.smooth3(p1)
        #c2 = self.latlayer3(c2)
        #c2 = self.conv(c2)
        #c3 = self.latlayer2(c3)
        #c3 = self.conv(c3)
        #c4 = self.latlayer1(c4)

        #x = self._upsample_add_(c2,c3,c4)
        #c4 = self.conv(c4)
        #c5 = self.toplayer(c5)
        x = self._upsample_cat(p1,p2,p3,p4,p5)

        #x = self.conv(x)
        #x = self.out_conv(x)
        h, w = c5.size()[2:]
        '''
        if self.train:
            x = F.interpolate(x, size=(h, w), mode='bilinear',align_corners=True)
        else:
            x = F.interpolate(x, size=(H // self.scale, W // self.scale), mode='bilinear',align_corners=True)
        if self.onnx:
            x = F.sigmoid(x)
        '''

        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear',align_corners=True) + y
    def _upsample_add_(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear',align_corners=True) + y

    def _upsample_cat(self,p1, p2, p3, p4, c5):
        h, w = p2.size()[2:]
        p1 = F.interpolate(p1, size=(h, w), mode='bilinear', align_corners=True)
        p2 = F.interpolate(p2, size=(h, w), mode='bilinear',align_corners=True)
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear',align_corners=True)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear',align_corners=True)
        p5 = F.interpolate(c5, size=(h, w), mode='bilinear',align_corners=True)
        return p1 + p2 + p3 + p4 + p5


if __name__ == '__main__':
    pass