import torch
from auxiliary.my_utils import yellow_print
from model.model import EncoderDecoder
import torch.optim as optim
import numpy as np
import torch.nn as nn
from copy import deepcopy

class TrainerModel(object):
    def __init__(self):
        """
        This class creates the architectures and implements all trainer functions related to architecture.
        Author : Thibault Groueix 01.11.2019
        """
        super(TrainerModel, self).__init__()

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        if torch.cuda.is_available():
            self.opt.device = torch.device(f"cuda:{self.opt.multi_gpu[0]}")
        else:
            # Run on CPU
            self.opt.device = torch.device(f"cpu")

        self.network = EncoderDecoder(self.opt)
        self.network = nn.DataParallel(self.network, device_ids=self.opt.multi_gpu)

        self.reload_network()

    def reload_network(self):
        """
        Reload entire model or only decoder (atlasnet) depending on the options
        :return:
        """
        #self.opt.reload_model_path="network.pth"
        if self.opt.reload_model_path != "":
            yellow_print(f"Network weights loaded from  {self.opt.reload_model_path}!")
            # print(self.network.state_dict().keys())
            # print(torch.load(self.opt.reload_model_path).keys())

            state_dict = torch.load(self.opt.reload_model_path, map_location='cuda:0')
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            self.network.module.load_state_dict(new_state_dict)

        elif self.opt.reload_decoder_path != "":
            opt = deepcopy(self.opt)
            opt.SVR = False
            network = EncoderDecoder(opt)
            network = nn.DataParallel(network, device_ids=opt.multi_gpu)
            network.module.load_state_dict(torch.load(opt.reload_decoder_path, map_location='cuda:0'))
            self.network.module.decoder = network.module.decoder
            yellow_print(f"Network Decoder weights loaded from  {self.opt.reload_decoder_path}!")

        else:
            yellow_print("No network weights to reload!")

    def build_optimizer(self):
        """
        Create optimizer
        """
        if self.opt.train_only_encoder:
            # To train a resnet image encoder with a pre-trained atlasnet decoder.
            yellow_print("only train the Encoder")
            self.optimizer = optim.Adam(self.network.module.encoder.parameters(), lr=self.opt.lrate)
        else:
            self.optimizer = optim.Adam(self.network.module.parameters(), lr=self.opt.lrate)

        if self.opt.reload_optimizer_path != "":
            try:
                self.optimizer.load_state_dict(torch.load(self.opt.reload_optimizer_path, map_location='cuda:0'))
                # yellow_print(f"Reloaded optimizer {self.opt.reload_optimizer_path}")
            except:
                yellow_print(f"Failed to reload optimizer {self.opt.reload_optimizer_path}")

        # Set policy for warm-up if you use multiple GPUs
        self.next_learning_rates = []
        if len(self.opt.multi_gpu) > 1:
            self.next_learning_rates = np.linspace(self.opt.lrate, self.opt.lrate * len(self.opt.multi_gpu),
                                                   5).tolist()
            self.next_learning_rates.reverse()
