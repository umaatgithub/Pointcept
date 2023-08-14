import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from timm.models.layers import trunc_normal_

from .mink_layers import MinkConvBNRelu, MinkResBlock
from .swin3d_layers import GridDownsample, GridKNNDownsample, BasicLayer, Upsample
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset


@MODELS.register_module("Swin3D-v1m1-ssc")
class Swin3DUNetSSC(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 base_grid_size,
                 depths,
                 channels,
                 num_heads,
                 window_sizes,
                 quant_size,
                 drop_path_rate=0.2,
                 up_k=3,
                 num_layers=5,
                 stem_transformer=True,
                 down_stride=2,
                 upsample='linear',
                 knn_down=True,
                 cRSE='XYZ_RGB',
                 fp16_mode=0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        if knn_down:
            downsample = GridKNNDownsample
        else:
            downsample = GridDownsample

        self.cRSE = cRSE
        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
            )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(
                    in_channels=channels[0],
                    out_channels=channels[0]
                )
            )
            self.downsample = downsample(
                channels[0],
                channels[1],
                kernel_size=down_stride,
                stride=down_stride
            )
            self.layer_start = 1
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=channels[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                quant_size=quant_size,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample if i < num_layers - 1 else None,
                down_stride=down_stride if i == 0 else 2,
                out_channels=channels[i + 1] if i < num_layers - 1 else None,
                cRSE=cRSE,
                fp16_mode=fp16_mode) for i in range(self.layer_start, num_layers)])

        if 'attn' in upsample:
            up_attn = True
        else:
            up_attn = False

        self.upsamples = nn.ModuleList([
            Upsample(channels[i], channels[i - 1], num_heads[i - 1], window_sizes[i - 1], quant_size, attn=up_attn, \
                     up_k=up_k, cRSE=cRSE, fp16_mode=fp16_mode)
            for i in range(num_layers - 1, 0, -1)])

        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], num_classes)
        )
        self.num_classes = num_classes
        self.base_grid_size = base_grid_size
        self.init_weights()

    def forward(self, data_dict):
        discrete_coord = data_dict["discrete_coord"]
        feat = data_dict["feat"]
        coord = data_dict["coord"]
        color = data_dict["color"]
        normal = data_dict["normal"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)
        in_field = ME.TensorField(
            features=torch.cat([batch.unsqueeze(-1),
                                coord / self.base_grid_size,
                                color / 1.001,
                                normal / 1.001,
                                feat
                                ], dim=1),
            coordinates=torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=feat.device)

        breakpoint()
        sp = in_field.sparse()
        coords_sp = SparseTensor(
            features=sp.F[:, :10],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )
        sp = SparseTensor(
            features=sp.F[:, 10:],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )

        sp_stack = []
        coords_sp_stack = []
        #[213619, 9], [213619, 10]
        sp = self.stem_layer(sp)
        #[213619, 48], [213619, 10]
        
        if self.layer_start > 0: #x
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            ''' Debug
            # 0:[213619,  48], [213619, 10]
            # 1:[ 21922,  96], [ 21922, 10]
            # 2:[  5161, 192], [  5161, 10]
            # 3:[  1160, 384], [  1160, 10]
            # 4:[   251, 384], [   251, 10]
            '''
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            ''' Debug
            # 0:[213619,  48], [21922,  96], [21922, 10]
            # 1:[ 21922,  96], [ 5161, 192], [ 5161, 10]
            # 2:[  5161, 192], [ 1160, 384], [ 1160, 10]
            # 3:[  1160, 384], [  251, 384], [  251, 10]
            # 4:[   251, 384], [  251, 384], [  251, 10]
            '''
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down

        sp = sp_stack.pop()
        coords_sp = coords_sp_stack.pop()
        breakpoint()
        for i, upsample in enumerate(self.upsamples):
            sp_i = sp_stack.pop()
            coords_sp_i = coords_sp_stack.pop()
            ''' Debug
            # 0: torch.Size([251, 384]), torch.Size([251, 10]), torch.Size([1160, 384]), torch.Size([1160, 10])
            # 1: torch.Size([1160, 384]), torch.Size([1160, 10]), torch.Size([5161, 192]), torch.Size([5161, 10])
            # 2: torch.Size([5161, 192]), torch.Size([5161, 10]), torch.Size([21922, 96]), torch.Size([21922, 10])
            # 3: torch.Size([21922, 96]), torch.Size([21922, 10]), torch.Size([213619, 48]), torch.Size([213619, 10])
            '''
            sp = upsample(sp, coords_sp, sp_i, coords_sp_i)
            coords_sp = coords_sp_i
        
        # [213619, 48]
        output = self.classifier(sp.slice(in_field).F)
        # Dense Tensor [214618, 13]
        return output

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)