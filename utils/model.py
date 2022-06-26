import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models

import numpy as np
import os

#TODO: add updated nesting layer code [GB]
from NestingLayer import *

#TODO: remove notion of nesting start? afaik we never used nesting_start=4 in paper [GB]
nesting_start=3

class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

class Model():
    def __init__(self, distributed, gpu, nesting, single_head, fixed_feature, use_blurpool):
        super().__init__()
        self.distributed = distributed
        self.gpu = gpu
        self.nesting = nesting
        self.sh = single_head
        self.ff = fixed_feature
        self.use_blurpool = use_blurpool

    def setup_distributed(self, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        torch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def load_model(self, model, model_weights_disk):
        if os.path.isfile(model_weights_disk):
            print("=> loading checkpoint '{}'".format(model_weights_disk))
            if self.gpu is None:
                checkpoint = torch.load(model_weights_disk)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(self.gpu)
                checkpoint = torch.load(model_weights_disk, map_location=loc)
            model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}' "
                  .format(model_weights_disk))
        else:
            print("=> no model found at '{}'".format(model_weights_disk))

        return model

    # TODO: rename to MRL style based on updated train code [GB]
    def initModel(self):
        print("Model init: nesting=%d, sh=%d, ff=%d" %(self.nesting, self.sh, self.ff))
        model = models.resnet50(pretrained=True)
        nesting_list = [2**i for i in range(nesting_start, 12)] if self.nesting else None

        # Nesting/Fixed Feature Modification code block
        if self.nesting:
            ff= "Single Head" if self.sh else "Multi Head"
            print("Using Nesting of type - {}".format(ff))
            print("Nesting Starts from {}".format(2**nesting_start))
            if self.sh:
                model.fc =  SingleHeadNestedLinear(nesting_list, num_classes=1000)
            else:
                model.fc =  MultiHeadNestedLinear(nesting_list, num_classes=1000)
        elif self.ff != 2048:
            print(f"Using Fixed Features = {self.ff}")
            model.fc =  FixedFeatureLayer(self.ff, 1000)

        def apply_blurpool(mod: torch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if self.use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=torch.channels_last)
        model = model.to(self.gpu)

        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model