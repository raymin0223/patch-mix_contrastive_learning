import torch
import torchvision
from torch import Tensor
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models.efficientnet import _efficientnet_conf


# torchvision.__version__ == '0.11.0+cu113' (arguments for _efficientnet_conf are different in latest version)
class EfficientNet_B0(torchvision.models.efficientnet.EfficientNet):
    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf('efficientnet_b0', width_mult=1.0, depth_mult=1.0)
        super().__init__(inverted_residual_setting, 0.2)

        del self.classifier
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.final_feat_dim = 1280

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
                                              progress=progress)
        del state_dict['features.0.0.weight']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        #if len(missing) > 0:
            #raise AssertionError('Model code may be incorrect')

    def load_ssl_official_weights(self, progress=True):
        raise NotImplemented

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # x = self.classifier(x)

        return x


class EfficientNet_B1(torchvision.models.efficientnet.EfficientNet):
    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf('efficientnet_b1', width_mult=1.0, depth_mult=1.1)
        super().__init__(inverted_residual_setting, 0.2)

        del self.classifier
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.final_feat_dim = 1280

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
                                              progress=progress)
        del state_dict['features.0.0.weight']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        #if len(missing) > 0:
            #raise AssertionError('Model code may be incorrect')

    def load_ssl_official_weights(self, progress=True):
        raise NotImplemented

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # x = self.classifier(x)

        return x


class EfficientNet_B2(torchvision.models.efficientnet.EfficientNet):
    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf('efficientnet_b2', width_mult=1.1, depth_mult=1.2)
        super().__init__(inverted_residual_setting, 0.3)

        del self.classifier
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.final_feat_dim = 1408

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
                                              progress=progress)
        del state_dict['features.0.0.weight']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        #if len(missing) > 0:
            #raise AssertionError('Model code may be incorrect')

    def load_ssl_official_weights(self, progress=True):
        raise NotImplemented

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # x = self.classifier(x)

        return x
