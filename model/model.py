from model.atlasnet import Atlasnet
from model.model_blocks import PointNet
import torch.nn as nn
import model.resnet as resnet
from dataset.MVCNNLoader import *

class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        
        self.encoder1 = resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)
        self.encoder2 = MVCNN_Loader.cnet_2
        
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
    def encoderHybrid(x):
        x_pointscloud, x_img = x
        return torch.cat(self.encoder1(x_pointscloud), self.encoder2(x_img))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
