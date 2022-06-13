# coding=utf-8
from torchvision import models
import torch.nn as nn
import torch
from enhance.enhance_base import enhance_base

class enhance_vgg16(enhance_base):
    def __init__(self, args, imdb):
        decoder = self.get_decoder()
        vgg = self.get_vgg()
        fcs = self.get_fcs()
        print("using fcs...")
        vgg, decoder, fcs = self.load_param(args, vgg, decoder, fcs)
        self.encoders, self.decoders = self.splits(vgg,decoder)
        enhance_base.__init__(self, args, self.encoders, self.decoders, fcs, imdb)

    def splits(self,vgg,decoder):
        encoders=[]
        decoders=[]
        encoders.append(nn.Sequential(*list(vgg._modules.values())[:2]))
        encoders.append(nn.Sequential(*list(vgg._modules.values())[2:7]))
        encoders.append(nn.Sequential(*list(vgg._modules.values())[7:12]))
        encoders.append(nn.Sequential(*list(vgg._modules.values())[12:]))
        decoders.append(nn.Sequential(*list(decoder._modules.values())[:7]))
        decoders.append(nn.Sequential(*list(decoder._modules.values())[7:12]))
        decoders.append(nn.Sequential(*list(decoder._modules.values())[12:17]))
        decoders.append(nn.Sequential(*list(decoder._modules.values())[17:]))
        return encoders,decoders
    
    def get_fcs(self):
        fc1 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(inplace=True),nn.Linear(512,512))
        fc2 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(inplace=True),nn.Linear(512,512))
        return [fc1,fc2]

    def get_vgg(self):
        vgg = models.vgg16()
        vgg = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        vgg[4].ceil_mode = True
        vgg[9].ceil_mode = True
        vgg[16].ceil_mode = True
        return vgg

    def get_decoder(self):
        decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
        )
        return decoder

    def load_param(self, args, vgg, decoder, fcs):
        for param in vgg.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        for i in range(len(fcs)):
            for param in fcs[i].parameters():
                param.requires_grad = False
        decoder.load_state_dict(torch.load(args.decoder_path))
        vgg.load_state_dict(torch.load(args.encoder_path)['model'])
        vgg = nn.Sequential(*list(vgg.children())[:19])
        fcs[0].load_state_dict(torch.load(args.fc1))
        fcs[1].load_state_dict(torch.load(args.fc2))
        print("loaded encoder: "+args.encoder_path)
        print("loaded decoder: "+args.decoder_path)
        print("loaded fc1: "+args.fc1)
        print("loaded fc2: "+args.fc2)
        if args.random_style:
            print("random style is True")
            print("random style is True")
            print("random style is True")
        else:
            print("random style is False")
            print("random style is False")
            print("random style is False")
        return vgg, decoder, fcs
