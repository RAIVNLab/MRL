import torch.nn as nn
import math
import torch

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(FLAGS.width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class Block(nn.Module):
    def __init__(self, inp, outp, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]

        midp = [i // 4 for i in outp]
        layers = [
            SlimmableConv2d(inp, midp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(midp),
            nn.ReLU(inplace=True),

            SlimmableConv2d(midp, midp, 3, stride, 1, bias=False),
            SwitchableBatchNorm2d(midp),
            nn.ReLU(inplace=True),

            SlimmableConv2d(midp, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                SlimmableConv2d(inp, outp, 1, stride=stride, bias=False),
                SwitchableBatchNorm2d(outp),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        self.features = []
        # head
        assert input_size % 32 == 0

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[FLAGS.depth]
        feats = [64, 128, 256, 512] # These are just the planes in the standard code
        channels = [
            int(64 * width_mult) for width_mult in FLAGS.width_mult_list]
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 7, 2, 3,
                    bias=False),
                SwitchableBatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )
        )

        # body
        for stage_id, n in enumerate(self.block_setting):
            outp = [
                int(feats[stage_id] * width_mult * 4)
                for width_mult in FLAGS.width_mult_list]
            for i in range(n):
                if i == 0 and stage_id != 0:
                    self.features.append(Block(channels, outp, 2))
                else:
                    self.features.append(Block(channels, outp, 1))
                channels = outp

        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.outp = channels
        self.classifier = nn.Sequential(
            SlimmableLinear(
                self.outp,
                [num_classes for _ in range(len(self.outp))]
            )
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

"""config utilities for yml file."""
import os
import sys
import yaml

# singletone
FLAGS = None


class LoaderMeta(type):
    """Constructor for supporting `!include`.
    """
    def __new__(mcs, __name__, __bases__, __dict__):
        """Add include constructer to class."""
        # register the include constructor on the class
        cls = super().__new__(mcs, __name__, __bases__, __dict__)
        cls.add_constructor('!include', cls.construct_include)
        return cls


class Loader(yaml.Loader, metaclass=LoaderMeta):
    """YAML Loader with `!include` constructor.
    """
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

    def construct_include(self, node):
        """Include file referenced at node."""
        filename = os.path.abspath(
            os.path.join(self._root, self.construct_scalar(node)))
        extension = os.path.splitext(filename)[1].lstrip('.')
        with open(filename, 'r') as f:
            if extension in ('yaml', 'yml'):
                return yaml.load(f, Loader)
            else:
                return ''.join(f.readlines())


class AttrDict(dict):
    """Dict as attribute trick.

    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, list):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return.

        """
        yaml_dict = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables.

        """
        ret_str = []
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # treat as AttrDict above
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    """Config with yaml file.

    This class is used to config model hyper-parameters, global constants, and
    other settings with yaml file. All settings in yaml file will be
    automatically logged into file.

    Args:
        filename(str): File name.

    Examples:

        yaml file ``model.yml``::

            NAME: 'neuralgym'
            ALPHA: 1.0
            DATASET: '/mnt/data/imagenet'

        Usage in .py:

        >>> from neuralgym import Config
        >>> config = Config('model.yml')
        >>> print(config.NAME)
            neuralgym
        >>> print(config.ALPHA)
            1.0
        >>> print(config.DATASET)
            /mnt/data/imagenet

    """

    def __init__(self, filename=None, verbose=False):
        assert os.path.exists(filename), 'File {} not exist.'.format(filename)
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, Loader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        super(Config, self).__init__(cfg_dict)
        if verbose:
            print(' pi.cfg '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def get_model(weight_path, config_path):
	# weight_path="/home/gbhatt2/imagenet/slimmable/logs/s_resnet50_0.25_0.5_0.75_1.0.pt"  # this needs to be changed to wherever is the saved model. 
	# config_path="/home/gbhatt2/imagenet/slimmable/apps/s_resnet50_train_val.yml" # Similarly this will change. 
	global FLAGS
	FLAGS =  Config(config_path)
	model = Model()

	ckpt = torch.load(weight_path)['model']
	plain_ckpt={}
	for k in ckpt.keys():
	    plain_ckpt[k[7:]] = ckpt[k]

	model.load_state_dict(plain_ckpt)
	return model


