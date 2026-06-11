import torch

import torch.nn.functional as F

from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss

from dagr.model.networks.net import Net
from dagr.model.layers.spline_conv import SplineConvToDense
from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_subnetwork, voxel_size_to_params, postprocess_network_output, convert_to_evaluation_format, init_grid_and_stride, convert_to_training_format


class DAGR(YOLOX):
    def __init__(self, args, height, width):
        self.conf_threshold = 0.001
        self.nms_threshold = 0.65

        self.height = height
        self.width = width

        backbone = Net(args, height=height, width=width)
        head = GNNHead(num_classes=backbone.num_classes,
                       in_channels=backbone.out_channels,
                       in_channels_cnn=backbone.out_channels_cnn,
                       strides=backbone.strides,
                       pretrain_cnn=args.pretrain_cnn,
                       args=args)

        super().__init__(backbone=backbone, head=head)

        if "img_net_checkpoint" in args:
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)
            init_subnetwork(self, state_dict['ema'], "head.cnn_head.")

    def cache_luts(self, width, height, radius):
        M = 2 * float(int(radius * width + 2) / width)
        r = int(radius * width+1)
        self.backbone.conv_block1.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=r)
        self.backbone.conv_block1.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=r)

        rx, ry, M = voxel_size_to_params(self.backbone.pool1, height, width)
        self.backbone.layer2.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer2.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool2, height, width)
        self.backbone.layer3.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer3.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool3, height, width)
        self.backbone.layer4.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer4.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        self.head.stem1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.cls_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.reg_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.cls_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.reg_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.obj_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool4, height, width)
        self.backbone.layer5.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer5.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        if self.head.num_scales > 1:
            self.head.stem2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.cls_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.cls_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.obj_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()

        if self.training:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)

            if self.backbone.use_image:
                targets0 = convert_to_training_format(x.bbox0, x.bbox0_batch, x.num_graphs)
                targets = (targets, targets0)

            # gt_target inputs need to be [l cx cy w h] in pixels
            outputs = YOLOX.forward(self, x, targets)

            return outputs

        x.reset = reset

        outputs = YOLOX.forward(self, x)

        detections = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)

        ret = [detections]

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret.append(targets)

        return ret


class CNNHead(YOLOXHead):
    def forward(self, xin):
        outputs = dict(cls_output=[], reg_output=[], obj_output=[])

        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)

            outputs["cls_output"].append(self.cls_preds[k](cls_feat))
            outputs["reg_output"].append(self.reg_preds[k](reg_feat))
            outputs["obj_output"].append(self.obj_preds[k](reg_feat))

        return outputs


class GNNHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        in_channels_cnn=[256, 512, 1024],
        act="silu",
        depthwise=False,
        pretrain_cnn=False,
        args=None
    ):
        YOLOXHead.__init__(self, num_classes, args.yolo_stem_width, strides, in_channels, act, depthwise)

        self.pretrain_cnn = pretrain_cnn
        self.num_scales = args.num_scales
        self.use_image = args.use_image
        self.batch_size = args.batch_size
        self.no_events = args.no_events
        self.args = args

        self.in_channels = in_channels
        self.n_anchors = 1
        self.num_classes = num_classes

        n_reg = max(in_channels)
        self.stem1 = ConvBlock(in_channels=in_channels[0], out_channels=n_reg, args=args)
        self.cls_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.cls_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
        self.reg_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.reg_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
        self.obj_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        if self.num_scales > 1:
            self.stem2 = ConvBlock(in_channels=in_channels[1], out_channels=n_reg, args=args)
            self.cls_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.cls_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
            self.reg_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.reg_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
            self.obj_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        if self.use_image:
            self.cnn_head = CNNHead(num_classes=num_classes, strides=strides, in_channels=in_channels_cnn)

        # ============ 新增：双向门控模块（修复版）============
        self.use_bidirectional_gating = getattr(args, 'use_bidirectional_gating', False)
        
        if self.use_bidirectional_gating and self.use_image:
            from dagr.model.layers.gating import EventActivityGating, SpatialEventActivityGating
            
            gate_type = getattr(args, 'gate_type', 'global')
            
            # GNN特征通道数（stem输出）
            gnn_channels = n_reg  # max(in_channels)
            
            # CNN检测头输出的通道数
            # cls: num_classes, reg: 4, obj: 1
            hidden_dim = getattr(args, 'gate_hidden_dim', 64)
            gate_strength_init = getattr(args, 'gate_strength_init', 0.1)
            
            # Scale 1: 为cls/reg/obj分别创建门控
            if gate_type == 'global':
                self.gate_scale1_cls = EventActivityGating(
                    gnn_channels=gnn_channels,
                    cnn_channels=self.num_classes,
                    hidden_dim=hidden_dim,
                    gate_strength_init=gate_strength_init
                )
                self.gate_scale1_reg = EventActivityGating(
                    gnn_channels=gnn_channels,
                    cnn_channels=4,
                    hidden_dim=hidden_dim,
                    gate_strength_init=gate_strength_init
                )
                self.gate_scale1_obj = EventActivityGating(
                    gnn_channels=gnn_channels,
                    cnn_channels=1,
                    hidden_dim=hidden_dim,
                    gate_strength_init=gate_strength_init
                )
            elif gate_type == 'spatial':
                spatial_bins = getattr(args, 'gate_spatial_bins', (8, 8))
                self.gate_scale1_cls = SpatialEventActivityGating(
                    gnn_channels=gnn_channels,
                    cnn_channels=self.num_classes,
                    spatial_bins=spatial_bins,
                    hidden_dim=hidden_dim,
                    gate_strength_init=gate_strength_init
                )
                self.gate_scale1_reg = SpatialEventActivityGating(
                    gnn_channels=gnn_channels,
                    cnn_channels=4,
                    spatial_bins=spatial_bins,
                    hidden_dim=hidden_dim,
                    gate_strength_init=gate_strength_init
                )
                self.gate_scale1_obj = SpatialEventActivityGating(
                    gnn_channels=gnn_channels,
                    cnn_channels=1,
                    spatial_bins=spatial_bins,
                    hidden_dim=hidden_dim,
                    gate_strength_init=gate_strength_init
                )
            else:
                raise ValueError(f"Unknown gate_type: {gate_type}")
            
            # Scale 2: 如果有多尺度
            if self.num_scales > 1:
                if gate_type == 'global':
                    self.gate_scale2_cls = EventActivityGating(
                        gnn_channels=gnn_channels,
                        cnn_channels=self.num_classes,
                        hidden_dim=hidden_dim,
                        gate_strength_init=gate_strength_init
                    )
                    self.gate_scale2_reg = EventActivityGating(
                        gnn_channels=gnn_channels,
                        cnn_channels=4,
                        hidden_dim=hidden_dim,
                        gate_strength_init=gate_strength_init
                    )
                    self.gate_scale2_obj = EventActivityGating(
                        gnn_channels=gnn_channels,
                        cnn_channels=1,
                        hidden_dim=hidden_dim,
                        gate_strength_init=gate_strength_init
                    )
                elif gate_type == 'spatial':
                    self.gate_scale2_cls = SpatialEventActivityGating(
                        gnn_channels=gnn_channels,
                        cnn_channels=self.num_classes,
                        spatial_bins=spatial_bins,
                        hidden_dim=hidden_dim,
                        gate_strength_init=gate_strength_init
                    )
                    self.gate_scale2_reg = SpatialEventActivityGating(
                        gnn_channels=gnn_channels,
                        cnn_channels=4,
                        spatial_bins=spatial_bins,
                        hidden_dim=hidden_dim,
                        gate_strength_init=gate_strength_init
                    )
                    self.gate_scale2_obj = SpatialEventActivityGating(
                        gnn_channels=gnn_channels,
                        cnn_channels=1,
                        spatial_bins=spatial_bins,
                        hidden_dim=hidden_dim,
                        gate_strength_init=gate_strength_init
                    )
            
            # 门控信息缓存
            self.gate_info_scale1 = {}
            self.gate_info_scale2 = {}
            self.current_epoch = 0
            
            print(f"\n[GNNHead] Bidirectional gating enabled:")
            print(f"  Gate type: {gate_type}")
            print(f"  GNN channels (stem output): {gnn_channels}")
            print(f"  CNN output channels: cls={self.num_classes}, reg=4, obj=1")
            print(f"  Hidden dim: {hidden_dim}")
            print(f"  Strength init: {gate_strength_init}")
            if self.num_scales > 1:
                print(f"  Scales: 2 (both with independent gates)")
        # ====================================================

        self.use_l1 = False
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        self.grid_cache = None
        self.stride_cache = None
        self.cache = []

    def should_apply_gating(self):
        """
        判断当前是否应该应用门控
        支持warmup策略：前N个epoch逐渐增强门控
        """
        if not self.use_bidirectional_gating:
            return False
        
        # 如果在eval模式且配置了不使用门控，则返回False
        if not self.training and not getattr(self.args, 'use_gating_in_eval', False):
            return False
        
        # Warmup策略
        warmup_epochs = getattr(self.args, 'gate_warmup_epochs', 0)
        if warmup_epochs > 0 and self.training:
            if self.current_epoch < warmup_epochs:
                # 线性增加门控强度
                alpha = self.current_epoch / warmup_epochs
                
                # 动态调整gate_strength
                target_strength = getattr(self.args, 'gate_strength_init', 0.1)
                for attr in ['gate_scale1_cls', 'gate_scale1_reg', 'gate_scale1_obj',
                             'gate_scale2_cls', 'gate_scale2_reg', 'gate_scale2_obj']:
                    gate_module = getattr(self, attr, None)
                    if gate_module is not None:
                        gate_module.gate_strength.data.fill_(alpha * target_strength)
        
        return True

    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred, batch_size, cache):
        x = stem(x)
        
        # 保存stem输出用于门控（关键修改）
        stem_output = shallow_copy(x)

        cls_feat = cls_conv(shallow_copy(x))
        reg_feat = reg_conv(x)

        # we need to provide the batchsize, since sometimes it cannot be found from the data, especially when nodes=0
        cls_output = cls_pred(cls_feat, batch_size=batch_size)
        reg_output = reg_pred(shallow_copy(reg_feat), batch_size=batch_size)
        obj_output = obj_pred(reg_feat, batch_size=batch_size)

        return cls_output, reg_output, obj_output, stem_output

    def apply_gating_to_cnn_output(self, cnn_output, gnn_stem_data, scale_idx):
        """
        应用门控调制CNN输出
        为cls/reg/obj三个输出分别应用独立的门控
        
        Args:
            cnn_output: dict with keys ['cls_output', 'reg_output', 'obj_output']
                - cls_output: (B, num_classes, H, W)
                - reg_output: (B, 4, H, W)
                - obj_output: (B, 1, H, W)
            gnn_stem_data: Data对象，包含stem之后的GNN特征
            scale_idx: 0 or 1，表示scale1或scale2
        
        Returns:
            modulated_output: dict with same keys as cnn_output
            gate_info: 门控信息dict
        """
        if not self.should_apply_gating():
            return cnn_output, {'gate_applied': False, 'reason': 'disabled'}
        
        # 提取GNN节点特征
        if gnn_stem_data is None or not hasattr(gnn_stem_data, 'x'):
            return cnn_output, {'gate_applied': False, 'reason': 'no_gnn_data'}
        
        gnn_node_features = gnn_stem_data.x  # (N_nodes, C_gnn)
        
        # 选择对应scale的门控模块
        if scale_idx == 0:
            gate_cls = self.gate_scale1_cls
            gate_reg = self.gate_scale1_reg
            gate_obj = self.gate_scale1_obj
        else:
            gate_cls = self.gate_scale2_cls
            gate_reg = self.gate_scale2_reg
            gate_obj = self.gate_scale2_obj
        
        # 为每个输出应用对应的门控
        modulated_output = {}
        
        # 根据gate_type决定传入参数
        gate_type = getattr(self.args, 'gate_type', 'global')
        
        if gate_type == 'spatial':
            # 空间门控需要位置信息
            modulated_output['cls_output'], info_cls = gate_cls(
                cnn_output['cls_output'],
                gnn_stem_data,  # 传入完整Data对象
                update=True
            )
            modulated_output['reg_output'], info_reg = gate_reg(
                cnn_output['reg_output'],
                gnn_stem_data,
                update=True
            )
            modulated_output['obj_output'], info_obj = gate_obj(
                cnn_output['obj_output'],
                gnn_stem_data,
                update=True
            )
        else:
            # 全局门控只需要特征
            modulated_output['cls_output'], info_cls = gate_cls(
                cnn_output['cls_output'],
                gnn_node_features,
                update=True
            )
            modulated_output['reg_output'], info_reg = gate_reg(
                cnn_output['reg_output'],
                gnn_node_features,
                update=True
            )
            modulated_output['obj_output'], info_obj = gate_obj(
                cnn_output['obj_output'],
                gnn_node_features,
                update=True
            )
        
        # 合并gate info
        gate_info = {
            'gate_applied': info_cls.get('gate_applied', False),
            'scale': scale_idx,
            'cls_gate': {
                'strength': info_cls.get('gate_strength', 0),
                'avg_value': info_cls.get('avg_gate_value', 0.5),
            },
            'reg_gate': {
                'strength': info_reg.get('gate_strength', 0),
                'avg_value': info_reg.get('avg_gate_value', 0.5),
            },
            'obj_gate': {
                'strength': info_obj.get('gate_strength', 0),
                'avg_value': info_obj.get('avg_gate_value', 0.5),
            },
            'num_nodes': info_cls.get('num_nodes', 0)
        }
        
        return modulated_output, gate_info

    def forward(self, xin: Data, labels=None, imgs=None):
        # for events + image outputs
        hybrid_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])
        image_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])

        if self.use_image:
            xin, image_feat = xin

            if labels is not None:
                if self.use_image:
                    labels, image_labels = labels

            # resize image, and process with CNN
            image_feat = [torch.nn.functional.interpolate(f, o) for f, o in zip(image_feat, self.output_sizes)]
            out_cnn = self.cnn_head(image_feat)

            # collect outputs from image alone, so the image network also learns to detect on its own.
            for k in [0, 1]:
                self.collect_outputs(out_cnn["cls_output"][k],
                                     out_cnn["reg_output"][k],
                                     out_cnn["obj_output"][k],
                                     k, self.strides[k], ret=image_out)

        batch_size = len(out_cnn["cls_output"][0]) if self.use_image else self.batch_size
        
        # ============ Scale 1 处理 ============
        cls_output, reg_output, obj_output, stem_output1 = self.process_feature(
            xin[0], self.stem1, self.cls_conv1, self.reg_conv1,
            self.cls_pred1, self.reg_pred1, self.obj_pred1, 
            batch_size=batch_size, cache=self.cache
        )

        if self.use_image:
            # ============ 应用门控（修复版）============
            if self.use_bidirectional_gating and hasattr(self, 'gate_scale1_cls'):
                # 构建CNN输出dict
                cnn_out_dict = {
                    'cls_output': out_cnn["cls_output"][0],  # (B, 2, H, W)
                    'reg_output': out_cnn["reg_output"][0],  # (B, 4, H, W)
                    'obj_output': out_cnn["obj_output"][0]   # (B, 1, H, W)
                }
                
                # 应用门控调制
                modulated_cnn_out, gate_info = self.apply_gating_to_cnn_output(
                    cnn_out_dict,
                    stem_output1,
                    scale_idx=0
                )
                
                # 保存门控信息用于logging
                self.gate_info_scale1 = gate_info
                
                # 使用调制后的CNN输出
                if gate_info.get('gate_applied', False):
                    cls_output[:batch_size] += modulated_cnn_out['cls_output']
                    reg_output[:batch_size] += modulated_cnn_out['reg_output']
                    obj_output[:batch_size] += modulated_cnn_out['obj_output']
                else:
                    # 退化为原始逻辑
                    cls_output[:batch_size] += out_cnn["cls_output"][0].detach()
                    reg_output[:batch_size] += out_cnn["reg_output"][0].detach()
                    obj_output[:batch_size] += out_cnn["obj_output"][0].detach()
            else:
                # 原始逻辑：不使用门控
                cls_output[:batch_size] += out_cnn["cls_output"][0].detach()
                reg_output[:batch_size] += out_cnn["reg_output"][0].detach()
                obj_output[:batch_size] += out_cnn["obj_output"][0].detach()
            # =========================================

        self.collect_outputs(cls_output, reg_output, obj_output, 0, self.strides[0], ret=hybrid_out)

        # ============ Scale 2 处理 ============
        if self.num_scales > 1:
            cls_output, reg_output, obj_output, stem_output2 = self.process_feature(
                xin[1], self.stem2, self.cls_conv2, self.reg_conv2, 
                self.cls_pred2, self.reg_pred2, self.obj_pred2, 
                batch_size=batch_size, cache=self.cache
            )
            
            if self.use_image:
                # ============ Scale 2 门控 ============
                if self.use_bidirectional_gating and hasattr(self, 'gate_scale2_cls'):
                    cnn_out_dict = {
                        'cls_output': out_cnn["cls_output"][1],
                        'reg_output': out_cnn["reg_output"][1],
                        'obj_output': out_cnn["obj_output"][1]
                    }
                    
                    modulated_cnn_out, gate_info = self.apply_gating_to_cnn_output(
                        cnn_out_dict,
                        stem_output2,
                        scale_idx=1
                    )
                    
                    self.gate_info_scale2 = gate_info
                    
                    if gate_info.get('gate_applied', False):
                        cls_output[:batch_size] += modulated_cnn_out['cls_output']
                        reg_output[:batch_size] += modulated_cnn_out['reg_output']
                        obj_output[:batch_size] += modulated_cnn_out['obj_output']
                    else:
                        cls_output[:batch_size] += out_cnn["cls_output"][1].detach()
                        reg_output[:batch_size] += out_cnn["reg_output"][1].detach()
                        obj_output[:batch_size] += out_cnn["obj_output"][1].detach()
                else:
                    # 原始逻辑
                    cls_output[:batch_size] += out_cnn["cls_output"][1].detach()
                    reg_output[:batch_size] += out_cnn["reg_output"][1].detach()
                    obj_output[:batch_size] += out_cnn["obj_output"][1].detach()
                # ======================================

            self.collect_outputs(cls_output, reg_output, obj_output, 1, self.strides[1], ret=hybrid_out)

        if self.training:
            # if we are only training the image detectors (pretraining),
            # we only need to minimize the loss at detections from the image branch.
            if self.use_image:
                losses_image = self.get_losses(
                    imgs,
                    image_out['x_shifts'],
                    image_out['y_shifts'],
                    image_out['expanded_strides'],
                    image_labels,
                    torch.cat(image_out['outputs'], 1),
                    image_out['origin_preds'],
                    dtype=image_out['x_shifts'][0].dtype,
                )

                if not self.pretrain_cnn:
                    losses_events  = self.get_losses(
                    imgs,
                    hybrid_out['x_shifts'],
                    hybrid_out['y_shifts'],
                    hybrid_out['expanded_strides'],
                    labels,
                    torch.cat(hybrid_out['outputs'], 1),
                    hybrid_out['origin_preds'],
                    dtype=xin[0].x.dtype,
                )

                    losses_image = list(losses_image)
                    losses_events = list(losses_events)

                    for i in range(5):
                        losses_image[i] = losses_image[i] + losses_events[i]

                return losses_image
            else:
                return self.get_losses(
                    imgs,
                    hybrid_out['x_shifts'],
                    hybrid_out['y_shifts'],
                    hybrid_out['expanded_strides'],
                    labels,
                    torch.cat(hybrid_out['outputs'], 1),
                    hybrid_out['origin_preds'],
                    dtype=xin[0].x.dtype,
                )
        else:
            out = image_out['outputs'] if self.no_events else hybrid_out['outputs']

            self.hw = [x.shape[-2:] for x in out]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in out], dim=2).permute(0, 2, 1)

            return self.decode_outputs(outputs, dtype=out[0].type())

    def collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, output.type())
            ret['x_shifts'].append(grid[:, :, 0])
            ret['y_shifts'].append(grid[:, :, 1])
            ret['expanded_strides'].append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output))
        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

        ret['outputs'].append(output)

    def decode_outputs(self, outputs, dtype):
        if self.grid_cache is None:
            self.grid_cache, self.stride_cache = init_grid_and_stride(self.hw, self.strides, dtype)

        outputs[..., :2] = (outputs[..., :2] + self.grid_cache) * self.stride_cache
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.stride_cache
        return outputs