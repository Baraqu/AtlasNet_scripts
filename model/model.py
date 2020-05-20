from model.atlasnet import Atlasnet
from model.model_blocks import PointNet
import torch.nn as nn
import torch
import model.resnet as resnet
from model.MVCNNLoader import my_MVCNN as MVCNN

class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        
        self.encoder1 = PointNet(nlatent=opt.bottleneck_size)
        # initialize MVCNN
        self.encoder2 = MVCNN()
        
        # if opt.SVR:
        #     self.encoder = resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)
        # else:
        #     self.encoder = PointNet(nlatent=opt.bottleneck_size)

        self.decoder = Atlasnet(opt)
        self.to(opt.device)

        if not opt.SVR:
            self.apply(weights_init)  # initialization of the weights
        self.eval()

    def forward(self, x, train=True):
        return self.decoder(self.encoderHybrid(x), train=train)

    def generate_mesh(self, x):
        return self.decoder.generate_mesh(self.encoderHybrid(x))
    
    # data comes from make_network_input

    def encoderHybrid(self, x):
        x_img, x_pointscloud = x
        feature_points = self.encoder1(x_pointscloud)
        feature_images = self.encoder2(x_img)
        print(feature_points.size(), feature_images.size())
        return torch.cat((feature_points, feature_images), dim=1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
