import random
import torch
from torchvision import transforms
import models
import vin
# import importlib
# foobar = importlib.import_module("semantic-segmentation")
# from foobar.semseg.models.backbones.convnext.py import ConvNeXt

class DQNPolicy:
    def __init__(self, cfg, action_space, train=False, random_seed=None):
        self.cfg = cfg
        self.action_space = action_space
        self.train = train

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = self.build_network()
        self.transform = transforms.ToTensor()

        # Resume from checkpoint if applicable
        if self.cfg.checkpoint_path is not None:
            model_checkpoint = torch.load(self.cfg.model_path, map_location=self.device)
            self.policy_net.load_state_dict(model_checkpoint['state_dict'])
            if self.train:
                self.policy_net.train()
            else:
                self.policy_net.eval()
            print("=> loaded model '{}'".format(self.cfg.model_path))

        if random_seed is not None:
            random.seed(random_seed)

    def build_network(self):
        raise NotImplementedError

    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def step(self, state, exploration_eps=None, debug=False):
        if exploration_eps is None:
            exploration_eps = self.cfg.final_exploration
        state = self.apply_transform(state).to(self.device)
        with torch.no_grad():
            self.policy_net.eval()
            output = self.policy_net(state).squeeze(0)
            self.policy_net.train()
        if random.random() < exploration_eps:
            action = random.randrange(self.action_space)
        else:
            action = output.view(1, -1).max(1)[1].item()
        info = {}
        if debug:
            info['output'] = output.squeeze(0)
        return action, info

class SteeringCommandsPolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            models.SteeringCommandsDQN(num_input_channels=self.cfg.num_input_channels, num_output_channels=self.action_space)
        ).to(self.device)

class DenseActionSpacePolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            models.DenseActionSpaceDQN(num_input_channels=self.cfg.num_input_channels)
        ).to(self.device)

class VINPolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            vin.VIN(l_i=self.cfg.num_input_channels, l_h=150, l_q=12)
        ).to(self.device)

class SuperVINPolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            vin.SuperVIN(l_i=self.cfg.num_input_channels, l_h=150, l_q=12)
        ).to(self.device)

class UNETPolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=4, out_channels=1, init_features=32, pretrained=False, trust_repo=True)
        ).to(self.device)

from ConvNeXt.semantic_segmentation.backbone.convnext import ConvNeXt
from mmseg.models.decode_heads.uper_head import UPerHead

class Stupid(torch.nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.backbone = ConvNeXt(
                in_chans=3,
                depths=[3, 3, 9, 3], 
                dims=[96, 192, 384, 768], 
                drop_path_rate=0.4,
                layer_scale_init_value=1.0,
                out_indices=[0, 1, 2, 3],
            )
        self.decode_head=UPerHead(
                in_channels=[96, 192, 384, 768],
                num_classes=2,
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
            )

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x

class ConvNextPolicy(DQNPolicy):
    def build_network(self):
        return Stupid() \
        .to(self.device)
