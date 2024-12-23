from models.blocks import *
from models.losses import *
from models.helpers import *
from collections import OrderedDict
import copy

import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T

from loguru import logger

from utils.helpers import timing_start, timing_end


def create_VARCNNBackbone(cfg):
    return VARCNNBackbone(cfg)


def create_EEGConvNetBackbone(cfg):
    return EEGConvNetBackbone(cfg)


backbone_dict = {
    "resnet18": [models.resnet18, 512],
    "resnet34": [models.resnet34, 512],
    "resnet50": [models.resnet50, 2048],
    "varcnn": [create_VARCNNBackbone, 1800],
    "eegconvnet": [create_EEGConvNetBackbone, 64],
}


class TSEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dims = cfg.DATASET.CHANNELS
        self.output_dims = cfg.MODEL.ARGS.PROJECTION_DIM
        self.hidden_dims = cfg.MODEL.ARGS.HIDDEN_SIZE
        self.mask_mode = cfg.MODEL.ARGS.MASK_MODE
        self.depth = cfg.MODEL.ARGS.DEPTH
        self.input_fc = nn.Linear(self.input_dims, self.hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            self.hidden_dims,
            [self.hidden_dims] * self.depth + [self.output_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(
        self, x, mask=None, return_embedding=True, return_projection=False
    ):  # x: B x T x input_dims
        x = x.transpose(1, 2).squeeze()
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"

        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)
        if self.repr_dropout is not None:
            x = self.repr_dropout(x)  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        x = F.max_pool1d(x.transpose(1, 2).contiguous(), kernel_size=x.size(1)).transpose(
            1, 2
        )
        x = x.squeeze()
        # normalize feature
        x = F.normalize(x, dim=-1)

        return x


# MLP class for projector and predictor
def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None, simple=False):
    if simple:
        return nn.Sequential(
            nn.Linear(dim, projection_size),
        )
    else:
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )


def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None, simple=False):
    if simple:
        return nn.Linear(dim, projection_size, bias=False)
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False),
    )


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(
        self,
        net,
        projection_size,
        projection_hidden_size,
        layer=-2,
        use_simsiam_mlp=False,
        sync_batchnorm=None,
        simple=False,
    ):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

        self.simple = simple

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(
            dim,
            self.projection_size,
            self.projection_hidden_size,
            sync_batchnorm=self.sync_batchnorm,
            simple=self.simple,
        )
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x, return_projection=True):
        representations = self.get_representation(x)
        if type(self.net).__name__ == "EEGConvNetBackbone":
            representation = representations[0]
            inv_representation = representations[1]
            acs_representation = representations[2]
            # logger.debug(representation.shape)
            if not return_projection:
                return representation, inv_representation, acs_representation
        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


# main class
class BYOL(nn.Module):
    def __init__(
        self,
        cfg,
        # net,
        # feature_size,
        # projection_size=256,
        # projection_hidden_size=4096,
        # moving_average_decay=0.99,
        # use_momentum=True,
        # sync_batchnorm=None,
    ):
        super().__init__()
        feature_size = cfg.DATASET.POINTS
        channels = cfg.DATASET.CHANNELS
        hidden_layer = cfg.MODEL.ARGS.HIDDEN_LAYER
        projection_size = cfg.MODEL.ARGS.PROJECTION_DIM
        projection_hidden_size = cfg.MODEL.ARGS.PROJECTION_HIDDEN_SIZE
        moving_average_decay = cfg.MODEL.ARGS.TAU_BASE
        use_momentum = cfg.MODEL.ARGS.USE_MOMENTUM
        sync_batchnorm = None
        simple = False
        if "resnet" in cfg.MODEL.ARGS.BACKBONE:

            self.net = backbone_dict[cfg.MODEL.ARGS.BACKBONE][0](pretrained=False)
            self.net.conv1 = nn.Conv2d(
                channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        else:
            self.net = backbone_dict[cfg.MODEL.ARGS.BACKBONE][0](cfg)
            # simple = True

        self.online_encoder = NetWrapper(
            self.net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
            sync_batchnorm=sync_batchnorm,
            simple=simple,
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size, simple=simple
        )

        # get device of network and make wrapper same device
        device = get_module_device(self.net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(
            torch.randn(1, channels, feature_size, 1, device=device),
            torch.randn(1, channels, feature_size, 1, device=device),
        )
        print(self.online_encoder)

    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
            self.use_momentum
        ), "you do not need to update the moving average, since you have turned off momentum for the target encoder"
        assert self.target_encoder is not None, "target encoder has not been created yet"
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(
        self, batch_view_1, batch_view_2, return_embedding=False, return_projection=True
    ):
        # assert not (
        #     self.training and batch_view_1.shape[0] == 1
        # ), "you must have greater than 1 sample when training, due to the batchnorm in the projection layer"

        if return_embedding:
            return self.online_encoder(batch_view_1, return_projection=return_projection)

        views = torch.cat((batch_view_1, batch_view_2), dim=0)

        online_projections, _ = self.online_encoder(views)
        # logger.debug(online_projections.shape)

        online_predictions = self.online_predictor(online_projections)
        # logger.debug(online_predictions.shape)

        online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)

        # logger.debug(online_pred_one.shape)
        with torch.no_grad():
            target_encoder = (
                self._get_target_encoder() if self.use_momentum else self.online_encoder
            )

            target_projections, _ = target_encoder(views)
            target_projections = target_projections.detach()

            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)
            # logger.debug(target_proj_one.shape)

        return (
            online_pred_one,
            online_pred_two,
            target_proj_one,
            target_proj_two,
        )


class BaseNet(nn.Module):
    def __init__(self, cfg):
        super(BaseNet, self).__init__()
        self.cfg = cfg

    def forward(self, x):
        raise NotImplementedError("Subclass must implement forward method")

    def compute_loss(self, output, target):
        criterion = nn.CrossEntropyLoss()
        return criterion(output, target)

    def get_learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class VARCNN(BaseNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        meg_channels = cfg.DATASET.CHANNELS
        points_length = cfg.DATASET.POINTS
        num_classes = cfg.DATASET.NUM_CLASSES

        max_pool = cfg.MODEL.ARGS.MAX_POOL
        sources_channels = cfg.MODEL.ARGS.SOURCE_CHANNELS

        Conv = VARConv

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("transpose0", Transpose(1, 2)),
                    ("Spatial", nn.Linear(meg_channels, sources_channels)),
                    ("transpose1", Transpose(1, 2)),
                    (
                        "Temporal_VAR",
                        Conv(
                            in_channels=sources_channels,
                            out_channels=sources_channels,
                            kernel_size=7,
                        ),
                    ),
                    ("unsqueeze", Unsqueeze(-3)),
                    ("active", nn.ReLU()),
                    ("pool", nn.MaxPool2d((1, max_pool), (1, max_pool))),
                    ("view", TensorView()),
                    ("dropout", nn.Dropout(p=0.5)),
                    (
                        "linear",
                        nn.Linear(sources_channels * int(points_length / 2), num_classes),
                    ),
                ]
            )
        )

    def forward(self, x, target=None):
        preds = self.net(x)
        loss = self.compute_loss(preds, target)
        return preds, loss


class VARCNNBackbone(BaseNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        meg_channels = cfg.DATASET.CHANNELS
        points_length = cfg.DATASET.POINTS
        num_classes = cfg.DATASET.NUM_CLASSES

        sources_channels = 360

        Conv = VARConv

        # refact nn.Sequential
        self.transpose0 = Transpose(1, 2)
        self.Spatial = nn.Linear(meg_channels, sources_channels)
        self.transpose1 = Transpose(1, 2)
        self.Temporal_VAR = Conv(
            in_channels=sources_channels,
            out_channels=sources_channels,
            kernel_size=7,
        )
        self.unsqueeze = Unsqueeze(-3)
        self.active = nn.LeakyReLU()
        self.pool = nn.MaxPool2d((1, 2), (1, 2))
        self.dist_inv_head = nn.Sequential(
            nn.Conv1d(sources_channels, sources_channels, 3, 1, 1), nn.LeakyReLU()
        )
        self.dist_acs_head = nn.Sequential(
            nn.Conv1d(sources_channels, sources_channels, 3, 1, 1), nn.LeakyReLU()
        )
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.view = TensorView()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, target=None):
        x = self.transpose0(x)
        x = x.squeeze()
        x = self.Spatial(x)
        x = self.transpose1(x)
        x = self.Temporal_VAR(x)
        # x = self.unsqueeze(x)
        x = self.active(x)
        x = self.pool(x)

        out_inv = self.dist_inv_head(x)
        out_inv = self.view(out_inv)

        out_acs = self.dist_acs_head(x)
        out_acs = self.view(out_acs)

        out = self.view(x)
        out = self.dropout(out)

        return out, out_inv, out_acs


class EEGConvNetBackbone(nn.Module):
    def __init__(self, cfg):
        super(EEGConvNetBackbone, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = ResidualBlock(64, 256, stride=1)
        self.block2 = ResidualBlock(256, 256, stride=1)
        self.block3 = ResidualBlock(256, 256, stride=1)
        # self.block1 = ResidualBlock(64, 256)
        # self.block2 = ResidualBlock(256, 256)
        # self.block3 = ResidualBlock(256, 256)
        self.block4 = ResidualBlock(256, 256)
        self.block5 = ResidualBlock(256, 256)
        self.view = TensorView()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dist_inv_head = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU())
        self.dist_acs_head = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU())
        # self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        out_inv = self.dist_inv_head(x)
        out_inv_g = self.avg_pool(out_inv)
        out_inv_g = self.view(out_inv_g)

        out_acs = self.dist_acs_head(x)
        out_acs_g = self.avg_pool(out_acs)
        out_acs_g = self.view(out_acs_g)

        out = self.avg_pool(x)
        out = self.view(out)

        return (x, out), (out_inv, out_inv_g), (out_acs, out_acs_g)


class SimCLR(BaseNet):
    def __init__(self, cfg):
        super(SimCLR, self).__init__(cfg)
        input_channels = cfg.DATASET.CHANNELS
        num_classes = cfg.DATASET.NUM_CLASSES

        backbone_name = cfg.MODEL.ARGS.BACKBONE
        projection_dim = cfg.MODEL.ARGS.PROJECTION_DIM
        n_features = cfg.MODEL.ARGS.N_FEATURES

        if "resnet" in backbone_name:
            self.backbone = backbone_dict[backbone_name][0](pretrained=False)
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.projection_head = nn.Sequential(
                nn.Linear(self.backbone.fc.in_features, self.backbone.fc.in_features),
                nn.ReLU(),
                nn.Linear(self.backbone.fc.in_features, projection_dim),
            )
            self.backbone.fc = nn.Identity()
        elif "varcnn" in backbone_name:
            self.backbone = backbone_dict[backbone_name][0](cfg)
            self.projection_head = nn.Sequential(
                nn.Linear(n_features, projection_dim),
            )
        print(self.backbone)

        # self.criterion = contrastive_loss(cfg)

        # self.fc = nn.Linear(projection_dim, num_classes)

    def forward(self, x_i, x_j, return_embedding=False, return_projection=True):
        if "resnet" in self.cfg.MODEL.ARGS.BACKBONE:
            x_i = torch.unsqueeze(x_i, -1)
            x_j = torch.unsqueeze(x_j, -1)
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)
        z_i = F.normalize(self.projection_head(h_i), dim=-1)
        z_j = F.normalize(self.projection_head(h_j), dim=-1)
        # logger.debug(z_i.shape)

        # loss = self.compute_loss(z_i, z_j)
        if return_embedding:
            return h_i
        return h_i, h_j, z_i, z_j


class LinearClassifier(nn.Module):
    def __init__(self, cfg):
        super(LinearClassifier, self).__init__()
        n_features = cfg.MODEL.ARGS.N_FEATURES
        n_classes = cfg.DATASET.NUM_CLASSES
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


class CurrentNetWrapper(nn.Module):
    def __init__(
        self,
        net,
        projection_size,
        projection_hidden_size,
        layer=-1,
        use_simsiam_mlp=False,
        sync_batchnorm=None,
        simple=False,
    ):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

        self.simple = simple

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(
            dim,
            self.projection_size,
            self.projection_hidden_size,
            sync_batchnorm=self.sync_batchnorm,
            simple=self.simple,
        )
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x, return_projection=True):
        representations = self.get_representation(x)
        representation = representations[0]
        inv4rec = representations[1][0]
        inv4clr = representations[1][1]
        acs = representations[2]
        # logger.debug(representation.shape)
        if not return_projection:
            return representation, inv4rec, acs

        projector_inv = self._get_projector(inv4clr)
        projection_inv = projector_inv(inv4clr)
        return projection_inv


class CurrentCLR(BaseNet):
    def __init__(
        self,
        cfg,
    ):
        super().__init__(cfg)
        feature_size = cfg.DATASET.POINTS
        channels = cfg.DATASET.CHANNELS
        hidden_layer = cfg.MODEL.ARGS.HIDDEN_LAYER
        projection_size = cfg.MODEL.ARGS.PROJECTION_DIM
        projection_hidden_size = cfg.MODEL.ARGS.PROJECTION_HIDDEN_SIZE
        moving_average_decay = cfg.MODEL.ARGS.TAU_BASE
        use_momentum = cfg.MODEL.ARGS.USE_MOMENTUM
        sync_batchnorm = None
        simple = False
        if "resnet" in cfg.MODEL.ARGS.BACKBONE:

            self.net = backbone_dict[cfg.MODEL.ARGS.BACKBONE][0](pretrained=False)
            self.net.conv1 = nn.Conv2d(
                channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        # elif "varcnn" in cfg.MODEL.ARGS.BACKBONE:
        else:
            self.net = backbone_dict[cfg.MODEL.ARGS.BACKBONE][0](cfg)
            # simple = True

        self.online_encoder = CurrentNetWrapper(
            self.net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
            sync_batchnorm=sync_batchnorm,
            simple=simple,
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size, simple=simple
        )

        # self.decoder = ConvGenerator(
        #     input_dim=projection_size,
        #     output_channels=channels,
        #     output_length=feature_size,
        # )
        # self.decoder = Generator(
        #     input_dim=projection_size,
        #     output_channels=channels,
        #     output_length=feature_size,
        # )
        # self.decoder = ConvDecoder(
        #     input_dim=projection_size*2,
        #     output_dim=feature_size
        # )
        self.decoder = ConvDecoder(filter_size=3, channels=channels, length=feature_size)
        self.cls_fc = nn.Linear(projection_size, cfg.DATASET.NUM_CLASSES)

        # regressive head
        self.pred_fc = nn.Sequential(
            nn.Linear(projection_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # get device of network and make wrapper same device
        device = get_module_device(self.net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(
            "clr",
            torch.randn(1, channels, feature_size, 1, device=device),
            torch.randn(1, channels, feature_size, 1, device=device),
        )
        print(self.net)

    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
            self.use_momentum
        ), "you do not need to update the moving average, since you have turned off momentum for the target encoder"
        assert self.target_encoder is not None, "target encoder has not been created yet"
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(
        self,
        step=None,
        clr_batch_view_1=None,
        clr_batch_view_2=None,
        rec_batch_view_spec=None,
        rec_batch_view_normal=None,
        cls_batch_view=None,
        pred_batch_view=None,
        return_embedding=False,
        return_projection=True,
    ):

        # assert not (
        #     self.training and batch_view_1.shape[0] == 1
        # ), "you must have greater than 1 sample when training, due to the batchnorm in the projection layer"

        if step == "clr":

            views = torch.cat((clr_batch_view_1, clr_batch_view_2), dim=0)

            online_projections = self.online_encoder(views, return_projection=True)
            online_predictions = self.online_predictor(online_projections)

            online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)

            with torch.no_grad():
                target_encoder = (
                    self._get_target_encoder()
                    if self.use_momentum
                    else self.online_encoder
                )

                target_projections = target_encoder(views)
                target_projections = target_projections.detach()

                target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)

            return (
                online_pred_one,
                online_pred_two,
                target_proj_one,
                target_proj_two,
            )

        elif step == "rec":
            representation, normal_inv_representation, normal_acs_representation = (
                self.online_encoder(rec_batch_view_normal, return_projection=False)
            )
            _, spec_inv_representation, spec_acs_representation = self.online_encoder(
                rec_batch_view_spec, return_projection=False
            )

            # representation4rec = representation[0]

            normal_acs_representation = normal_acs_representation[0]
            spec_acs_representation = spec_acs_representation[0]

            rec_spec_batch_one = self.decoder(
                spec_inv_representation, spec_acs_representation
            )

            rec_spec_batch_two = self.decoder(
                normal_inv_representation, spec_acs_representation
            )

            rec_normal_batch_one = self.decoder(
                normal_inv_representation, normal_acs_representation
            )

            rec_normal_batch_two = self.decoder(
                spec_inv_representation, normal_inv_representation
            )

            rec_spec_batch_one = rec_spec_batch_one.unsqueeze(-1)
            rec_spec_batch_two = rec_spec_batch_two.unsqueeze(-1)
            rec_normal_batch_one = rec_normal_batch_one.unsqueeze(-1)
            rec_normal_batch_two = rec_normal_batch_two.unsqueeze(-1)
            # rec_representation = rec_representation.unsqueeze(-1)
            return (
                rec_spec_batch_one,
                rec_spec_batch_two,
                rec_normal_batch_one,
                rec_normal_batch_two,
                # rec_representation,
                normal_inv_representation,
                spec_inv_representation,
                normal_acs_representation,
                spec_acs_representation,
            )

        elif step == "cls":
            _, _, cls_representation = self.online_encoder(
                cls_batch_view, return_projection=False
            )
            cls_logits = self.cls_fc(cls_representation)
            return cls_logits

        elif step == "pred":
            _, _, acs_representation = self.online_encoder(
                pred_batch_view, return_projection=False
            )
            pred_representation = acs_representation[1]

            pred_output = self.pred_fc(pred_representation)
            return pred_output

        else:
            # linear evaluation
            if return_embedding:
                return self.online_encoder(clr_batch_view_1, return_projection=False)
